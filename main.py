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
load_dotenv()

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# Konfigurasi MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/traffic_monitoring")
client = MongoClient(MONGO_URI)
db = client.get_database()
cameras_collection = db["cameras"]
stats_collection = db["stats_collection"]

# Muat Model YOLOv8
print("APP: Memuat model YOLOv8...")
model = YOLO('yolov8n.pt')
print("APP: Model berhasil dimuat.")

# Mapping class
CLASS_NAMES_YOLO = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
}

# --- Variabel Global untuk Streaming Video ---
output_frames = {} 
frame_locks = defaultdict(Lock)
streaming_threads = {}

# --- Fungsi Streaming Video ---

def stream_analyzed_video(camera_data):
    """
    Fungsi ini berjalan di thread untuk menganalisis video SECARA REALTIME untuk STREAMING.
    Tidak menyimpan statistik ke database.
    """
    camera_id = camera_data["camera_id"]
    rtsp_url = camera_data["rtsp_url"]
    line_position_ratio = camera_data.get("calibration_line_position", 0.5)

    track_history = defaultdict(lambda: [])
    
    while True:
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"APP Error ({camera_id}): Gagal membuka stream. Mencoba lagi...")
                time.sleep(10)
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            line_y = int(height * line_position_ratio)
            
            print(f"APP: Memulai stream realtime untuk kamera {camera_id}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"APP Stream ({camera_id}) terputus. Mencoba reconnect...")
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
                        class_name = CLASS_NAMES_YOLO.get(cls, "unknown")
                        cv2.rectangle(annotated_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255,0,0), 2)
                        cv2.putText(annotated_frame, f"ID:{track_id} {class_name}", (int(x-w/2), int(y-h/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    with frame_locks[camera_id]:
                        output_frames[camera_id] = buffer.tobytes()

        except Exception as e:
            print(f"APP Exception di thread kamera {camera_id}: {e}")
            time.sleep(5)
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

def generate_frames(camera_id):
    """Generator untuk mengambil frame dari thread dan mengirimkannya ke browser."""
    if camera_id not in streaming_threads:
        try:
            camera_data = cameras_collection.find_one({"camera_id": camera_id})
            if camera_data:
                thread = Thread(target=stream_analyzed_video, args=(camera_data,), daemon=True)
                streaming_threads[camera_id] = thread
                thread.start()
                print(f"APP: Thread streaming untuk {camera_id} dimulai.")
            else:
                print(f"APP Error: Data kamera {camera_id} tidak ditemukan di DB.")
        except Exception as e:
             print(f"APP: Gagal memulai thread streaming: {e}")

    while True:
        time.sleep(0.04)
        frame = None
        with frame_locks[camera_id]:
            frame = output_frames.get(camera_id)

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Rute Flask (Web Server) ---

@app.route('/')
def index():
    """Halaman utama yang menampilkan daftar kamera."""
    try:
        all_cameras = list(cameras_collection.find({}))
        for cam in all_cameras:
            cam['_id'] = str(cam['_id'])
        return render_template('index.html', cameras=all_cameras)
    except Exception as e:
        return f"Error koneksi database: {e}", 500

@app.route('/camera/<camera_id>')
def camera_detail(camera_id):
    """Halaman detail kamera yang menampilkan stream dan statistik."""
    camera = cameras_collection.find_one({"camera_id": camera_id})
    if not camera:
        return "Kamera tidak ditemukan", 404
    camera['_id'] = str(camera['_id'])
    return render_template('detail.html', camera=camera)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Endpoint untuk stream video MJPEG yang sudah dianalisis."""
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats/<camera_id>')
def get_stats(camera_id):
    """Endpoint API untuk mengambil data statistik dari MongoDB."""
    today_str = datetime.now().strftime("%Y-%m-%d")

    # --- INI BAGIAN QUERY YANG DIPERBAIKI ---
    # Kita mencari dokumen yang field 'camera_id'-nya cocok DAN
    # field 'date'-nya adalah tanggal hari ini.
    query = {
        "camera_id": camera_id,
        "date": today_str
    }
    print(f"APP: Mencari statistik untuk kamera {camera_id} pada {today_str}")
    stats_data = stats_collection.find_one(query)
    # -----------------------------------------
    print(f"APP: Data statistik ditemukan: {stats_data}")
    
    if stats_data:
        # Ambil data dari dokumen yang ditemukan
        response_data = {
            "total_car": stats_data.get("total_car", 0),
            "total_motorcycle": stats_data.get("total_motorcycle", 0),
            "total_bus": stats_data.get("total_bus", 0),
            "total_truck": stats_data.get("total_truck", 0),
            "average_speed": stats_data.get("average_speed", 0.0)
        }
        return jsonify(response_data)
    else:
        # Jika tidak ada data untuk hari ini, kembalikan nilai nol
        return jsonify({
            "total_car": 0, 
            "total_motorcycle": 0, 
            "total_bus": 0, 
            "total_truck": 0, 
            "average_speed": 0.0
        })

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5078, debug=True)