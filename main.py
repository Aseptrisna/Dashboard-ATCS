# app.py

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

# --- Konfigurasi dan Inisialisasi ---

# Muat variabel lingkungan
load_dotenv()

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# Konfigurasi MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/traffic_monitoring")
client = MongoClient(MONGO_URI)
db = client.get_database()
cameras_collection = db["cameras"]
stats_collection = db["daily_stats"]

# Muat Model YOLOv8
print("Memuat model YOLOv8...")
model = YOLO('yolov8n.pt')
print("Model berhasil dimuat.")

# Mapping class
CLASS_NAMES_MAP = {
    2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
}
CLASS_NAMES_YOLO = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
}

# --- Variabel Global untuk Komunikasi Antar Thread ---
processing_threads = {}
# Kunci: camera_id, Nilai: frame JPEG yang sudah di-encode
output_frames = {} 
# Lock untuk thread-safe access ke output_frames
frame_locks = defaultdict(Lock)


# --- Fungsi Logika Deteksi & Database ---

def update_daily_stats(camera_id, class_id, speed):
    """Memperbarui statistik harian di MongoDB."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    doc_id = f"{camera_id}_{today_str}"
    
    vehicle_type_key = CLASS_NAMES_MAP.get(class_id)
    if not vehicle_type_key:
        return

    stats_collection.update_one(
        {"_id": doc_id},
        {
            "$inc": {f"total_{vehicle_type_key}": 1},
            "$push": {"speeds": speed},
            "$setOnInsert": {
                "camera_id": camera_id, "date": today_str, "processed_at": time.time(),
                "total_car": 0, "total_motorcycle": 0, "total_bus": 0, "total_truck": 0
            }
        },
        upsert=True
    )

def process_camera_stream(camera_data):
    """
    Fungsi ini berjalan di background thread untuk setiap kamera.
    Membaca stream, melakukan deteksi, dan menyimpan hasilnya ke variabel global.
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
            if fps == 0: fps = 25

            print(f"Memulai pemrosesan untuk kamera {camera_id} di {rtsp_url}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Stream untuk kamera {camera_id} terputus. Mencoba reconnect...")
                    break

                annotated_frame = frame.copy()
                cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 0), 2)
                
                results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    classes = results[0].boxes.cls.int().cpu().tolist()
                    
                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((int(x), int(y)))
                        if len(track) > 30:
                            track.pop(0)

                        class_name = CLASS_NAMES_YOLO.get(cls, "unknown")
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f"ID:{track_id} {class_name}", (int(x-w/2), int(y-h/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                        if len(track) > 1 and (track[-2][1] < line_y and track[-1][1] >= line_y):
                            # Sederhanakan kalkulasi kecepatan untuk contoh ini
                            update_daily_stats(camera_id, cls, 0) # speed 0 for now
                            track_history.pop(track_id, None)
                
                # Simpan frame ke variabel global dengan lock
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    with frame_locks[camera_id]:
                        output_frames[camera_id] = buffer.tobytes()

        except Exception as e:
            print(f"Exception di thread kamera {camera_id}: {e}")
            time.sleep(5)
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()


# --- Logika Flask (Web Server) ---

def generate_frames(camera_id):
    """Generator untuk streaming frame dari variabel global ke response HTTP."""
    while True:
        time.sleep(0.04)
        frame = None
        with frame_locks[camera_id]:
            frame = output_frames.get(camera_id)

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Halaman utama."""
    try:
        all_cameras = list(cameras_collection.find({}))
        for cam in all_cameras:
            cam['_id'] = str(cam['_id'])
        return render_template('index.html', cameras=all_cameras)
    except Exception as e:
        return f"Error koneksi database: {e}", 500

@app.route('/camera/<camera_id>')
def camera_detail(camera_id):
    """Halaman detail kamera."""
    camera = cameras_collection.find_one({"camera_id": camera_id})
    if not camera:
        return "Kamera tidak ditemukan", 404
    camera['_id'] = str(camera['_id'])
    return render_template('detail.html', camera=camera)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Endpoint untuk stream video MJPEG."""
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats/<camera_id>')
def get_stats(camera_id):
    """Endpoint API untuk data statistik dari MongoDB."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    stats_data = stats_collection.find_one({"_id": f"{camera_id}_{today_str}"})
    
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
        return jsonify({
            "total_car": 0, "total_motorcycle": 0, "total_bus": 0, "total_truck": 0, "average_speed": 0
        })

def start_all_camera_processing():
    """Memulai thread pemrosesan untuk semua kamera di database."""
    print("Memulai pemrosesan latar belakang untuk semua kamera...")
    try:
        for camera in cameras_collection.find({}):
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
    # 1. Memulai thread pemrosesan di latar belakang
    start_all_camera_processing()
    
    # 2. Menjalankan aplikasi Flask
    # use_reloader=False penting agar thread background tidak dimulai dua kali
    app.run(host='0.0.0.0', port=1616, debug=True, use_reloader=False)