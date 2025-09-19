# ai_processor.py

import os
import cv2
import math
import time
import redis
from datetime import datetime
from collections import defaultdict
from threading import Thread
from pymongo import MongoClient
from dotenv import load_dotenv
from ultralytics import YOLO

# --- Konfigurasi dan Inisialisasi ---

# Muat variabel lingkungan dari file .env
load_dotenv()

# Konfigurasi MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/traffic_monitoring")
client = MongoClient(MONGO_URI)
db = client.get_database()
cameras_collection = db["cameras"]
stats_collection = db["daily_stats"]

# Konfigurasi Redis
# Pastikan Redis server sudah berjalan di localhost port 6379
try:
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), db=0)
    redis_client.ping()
    print("Berhasil terhubung ke Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"Gagal terhubung ke Redis: {e}")
    print("Pastikan server Redis Anda sedang berjalan.")
    exit()


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

# Variabel Global untuk threads
processing_threads = {}


# --- Fungsi Logika Deteksi ---

def update_daily_stats(camera_id, class_id, speed):
    """
    Memperbarui statistik harian di MongoDB.
    """
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
                "camera_id": camera_id,
                "date": today_str,
                "processed_at": time.time(),
                "total_car": 0, "total_motorcycle": 0, "total_bus": 0, "total_truck": 0
            }
        },
        upsert=True
    )

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
                        center = (int(x), int(y))
                        
                        track = track_history[track_id]
                        track.append(center)
                        if len(track) > 30:
                            track.pop(0)

                        class_name = CLASS_NAMES_YOLO.get(cls, "unknown")
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f"ID: {track_id} {class_name}", (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        if len(track) > 1:
                            prev_y = track[-2][1]
                            curr_y = track[-1][1]

                            if (prev_y < line_y and curr_y >= line_y) or (prev_y > line_y and curr_y <= line_y):
                                distance_pixels = math.sqrt((track[-1][0] - track[-2][0])**2 + (track[-1][1] - track[-2][1])**2)
                                pixel_to_meter = 0.1 
                                speed_kmh = (distance_pixels * pixel_to_meter * fps) * 3.6
                                update_daily_stats(camera_id, cls, speed_kmh)
                                track_history.pop(track_id, None)

                # Simpan frame yang sudah diolah ke REDIS
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    # Kunci Redis akan menjadi "frame:<camera_id>"
                    redis_client.set(f"frame:{camera_id}", buffer.tobytes())

        except Exception as e:
            print(f"Exception di thread kamera {camera_id}: {e}")
            time.sleep(5)
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()


def start_all_camera_processing():
    """
    Memulai thread pemrosesan untuk semua kamera yang ada di database.
    """
    print("Memulai pemrosesan latar belakang untuk semua kamera...")
    try:
        all_cameras = list(cameras_collection.find({}))
        if not all_cameras:
            print("Tidak ada kamera yang ditemukan di database.")
            return

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
    start_all_camera_processing()
    # Jaga agar skrip utama tetap berjalan selamanya
    while True:
        time.sleep(60)
        print("AI Processor masih berjalan...")