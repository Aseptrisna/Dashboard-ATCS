import os
import cv2
import math
import time
from datetime import datetime
from collections import defaultdict
from threading import Thread, Lock
from flask import Flask, render_template, Response, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
from ultralytics import YOLO
from bson import ObjectId
import json

# --- Konfigurasi dan Inisialisasi ---

# Muat variabel lingkungan dari file .env
load_dotenv()

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# Konfigurasi MongoDB
MONGO_URI = os.getenv("MONGO_URI", "/")
client = MongoClient(MONGO_URI)
db = client.get_database() # Menggunakan database dari URI
cameras_collection = db["cameras"]
stats_collection = db["daily_stats"] # Koleksi baru untuk statistik harian

# Muat Model YOLOv8
model = YOLO('yolov8n.pt')

# Mapping class ID ke nama yang lebih sederhana untuk statistik
CLASS_NAMES_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
# Mapping nama kelas dari model
CLASS_NAMES_YOLO = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
}

# Variabel Global untuk Background Processing
# Kunci: camera_id, Nilai: thread
processing_threads = {}
# Kunci: camera_id, Nilai: frame JPEG yang sudah di-encode
output_frames = {}
# Kunci: camera_id, Nilai: statistik terbaru
current_stats = {}
# Lock untuk thread-safe access ke output_frames
frame_locks = defaultdict(Lock)


# --- Fungsi Helper dan Logika Deteksi ---

def process_camera_stream(camera_data):
    """
    Fungsi ini berjalan di background thread untuk setiap kamera.
    Membaca stream RTSP, melakukan deteksi, dan menyimpan hasilnya.
    """
    camera_id = camera_data["camera_id"]
    rtsp_url = camera_data["rtsp_url"]
    line_position_ratio = camera_data.get("calibration_line_position", 0.5)

    track_history = defaultdict(lambda: [])
    
    while True:
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"Error: Gagal membuka stream untuk kamera {camera_id}. Mencoba lagi dalam 10 detik...")
                time.sleep(10)
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            line_y = int(height * line_position_ratio)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: fps = 25 # Default FPS jika tidak terbaca

            print(f"Memulai pemrosesan untuk kamera {camera_id} di {rtsp_url}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Stream untuk kamera {camera_id} terputus. Mencoba reconnect...")
                    break # Keluar dari loop dalam untuk reconnect

                annotated_frame = frame.copy()

                # Gambar garis kalibrasi
                cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 0), 2)

                # Lakukan tracking dengan YOLOv8
                results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    classes = results[0].boxes.cls.int().cpu().tolist()
                    
                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        x, y, w, h = box
                        center = (int(x), int(y))
                        
                        track = track_history[track_id]
                        track.append(center)
                        if len(track) > 30:
                            track.pop(0)

                        # Gambar bounding box dan label
                        class_name = CLASS_NAMES_YOLO.get(cls, "unknown")
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f"ID: {track_id} {class_name}", (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Cek jika kendaraan melewati garis
                        if len(track) > 1:
                            prev_y = track[-2][1]
                            curr_y = track[-1][1]

                            if (prev_y < line_y and curr_y >= line_y) or (prev_y > line_y and curr_y <= line_y):
                                # Hitung kecepatan (estimasi)
                                distance_pixels = math.sqrt((track[-1][0] - track[-2][0])**2 + (track[-1][1] - track[-2][1])**2)
                                # Asumsi kalibrasi: 10 piksel = 1 meter. Perlu disesuaikan.
                                pixel_to_meter = 0.1 
                                speed_kmh = (distance_pixels * pixel_to_meter * fps) * 3.6

                                update_daily_stats(camera_id, cls, speed_kmh)
                                
                                # Reset history untuk ID ini agar tidak dihitung berulang kali
                                track_history.pop(track_id, None)

                # Simpan frame yang sudah diolah untuk streaming
                with frame_locks[camera_id]:
                    ret, buffer = cv2.imencode('.jpg', annotated_frame)
                    output_frames[camera_id] = buffer.tobytes()

        except Exception as e:
            print(f"Exception di thread kamera {camera_id}: {e}")
            time.sleep(5) # Tunggu sebelum mencoba lagi
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()


def update_daily_stats(camera_id, class_id, speed):
    """
    Memperbarui statistik harian di MongoDB untuk satu kamera.
    Menggunakan `upsert=True` untuk membuat dokumen baru jika belum ada untuk hari itu.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    doc_id = f"{camera_id}_{today_str}"
    
    vehicle_type_key = CLASS_NAMES_MAP.get(class_id)
    if not vehicle_type_key:
        return # Abaikan jika bukan kelas yang kita pantau

    # Gunakan $inc untuk increment total kendaraan
    # Gunakan $push untuk menambah kecepatan ke array untuk dihitung rata-ratanya
    stats_collection.update_one(
        {"_id": doc_id},
        {
            "$inc": {f"total_{vehicle_type_key}": 1},
            "$push": {"speeds": speed},
            "$setOnInsert": {
                "camera_id": camera_id,
                "date": today_str,
                "processed_at": time.time(),
                "total_car": 0, "total_motorcycle": 0, "total_bus": 0, "total_truck": 0
            }
        },
        upsert=True
    )

def generate_frames(camera_id):
    """
    Generator untuk streaming frame dari background thread ke response HTTP.
    """
    while True:
        time.sleep(0.04) # Batasi FPS stream ke ~25fps
        with frame_locks[camera_id]:
            frame = output_frames.get(camera_id)

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Rute Flask ---

@app.route('/')
def index():
    """Halaman utama yang menampilkan daftar kamera."""
    try:
        # Ganti _id ObjectId menjadi string agar bisa di-render di template
        all_cameras = list(cameras_collection.find({}))
        for cam in all_cameras:
            cam['_id'] = str(cam['_id'])
        return render_template('index.html', cameras=all_cameras)
    except Exception as e:
        return f"Error koneksi database: {e}", 500

@app.route('/camera/<camera_id>')
def camera_detail(camera_id):
    """Halaman detail untuk satu kamera, menampilkan stream dan statistik."""
    try:
        camera = cameras_collection.find_one({"camera_id": camera_id})
        if not camera:
            return "Kamera tidak ditemukan", 404
        
        camera['_id'] = str(camera['_id'])
        return render_template('detail.html', camera=camera)
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Endpoint yang menyediakan stream video MJPEG."""
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats/<camera_id>')
def get_stats(camera_id):
    """Endpoint API untuk mendapatkan data statistik terbaru (via AJAX)."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    doc_id = f"{camera_id}_{today_str}"
    
    stats_data = stats_collection.find_one({"_id": doc_id})
    
    if stats_data:
        speeds = stats_data.get("speeds", [])
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        response_data = {
            "total_car": stats_data.get("total_car", 0),
            "total_motorcycle": stats_data.get("total_motorcycle", 0),
            "total_bus": stats_data.get("total_bus", 0),
            "total_truck": stats_data.get("total_truck", 0),
            "average_speed": round(avg_speed, 2)
        }
        return jsonify(response_data)
    else:
        # Jika belum ada data hari ini, kirim data kosong
        return jsonify({
            "total_car": 0, "total_motorcycle": 0, "total_bus": 0, "total_truck": 0, "average_speed": 0
        })

# --- Fungsi untuk Memulai Background Tasks ---

def start_all_camera_processing():
    """
    Memulai thread pemrosesan untuk semua kamera yang ada di database.
    """
    print("Memulai pemrosesan latar belakang untuk semua kamera...")
    try:
        all_cameras = list(cameras_collection.find({}))
        for camera in all_cameras:
            camera_id = camera.get("camera_id")
            if camera_id and camera_id not in processing_threads:
                thread = Thread(target=process_camera_stream, args=(camera,), daemon=True)
                processing_threads[camera_id] = thread
                thread.start()
                print(f"Thread untuk kamera {camera_id} telah dimulai.")
    except Exception as e:
        print(f"Gagal memulai pemrosesan latar belakang: {e}")


# --- Main Execution ---

if __name__ == '__main__':
    # Memulai thread pemrosesan di latar belakang sebelum server Flask berjalan
    start_all_camera_processing()
    
    # Menjalankan aplikasi Flask
    # Gunakan host='0.0.0.0' agar bisa diakses dari jaringan
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
