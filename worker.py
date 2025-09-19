# worker.py

import os
import cv2
import math  # DITAMBAHKAN
import time
from datetime import datetime
from collections import defaultdict
from threading import Thread
from pymongo import MongoClient
from dotenv import load_dotenv
from ultralytics import YOLO

# --- Konfigurasi dan Inisialisasi ---
load_dotenv()

# Konfigurasi MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/traffic_monitoring")
client = MongoClient(MONGO_URI)
db = client.get_database()
cameras_collection = db["cameras"]
stats_collection = db["daily_stats"]

# Muat Model YOLOv8
print("WORKER: Memuat model YOLOv8...")
model = YOLO('yolov8n.pt')
print("WORKER: Model berhasil dimuat.")

# Mapping class
CLASS_NAMES_MAP = {
    2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
}

processing_threads = {}

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
                "camera_id": camera_id, "date": today_str,
                "total_car": 0, "total_motorcycle": 0, "total_bus": 0, "total_truck": 0,
                "speeds": [] # DITAMBAHKAN: inisialisasi array speeds
            }
        },
        upsert=True
    )

def process_and_save_stats(camera_data):
    """
    Fungsi ini berjalan di background thread untuk setiap kamera.
    Menganalisis dan menyimpan statistik (termasuk kecepatan) ke DB.
    """
    camera_id = camera_data["camera_id"]
    rtsp_url = camera_data["rtsp_url"]
    line_position_ratio = camera_data.get("calibration_line_position", 0.5)

    track_history = defaultdict(lambda: [])
    
    while True:
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"WORKER Error ({camera_id}): Gagal membuka stream. Mencoba lagi dalam 10 detik...")
                time.sleep(10)
                continue

            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            line_y = int(height * line_position_ratio)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) # DITAMBAHKAN: ambil FPS
            if fps == 0: fps = 25 # DITAMBAHKAN: fallback jika FPS tidak terbaca
            
            print(f"WORKER: Memulai analisis statistik untuk kamera {camera_id}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"WORKER Stream ({camera_id}) terputus. Mencoba reconnect...")
                    break
                
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

                        if len(track) > 1:
                            prev_y = track[-2][1]
                            curr_y = track[-1][1]
                            
                            if (prev_y < line_y and curr_y >= line_y) or \
                               (prev_y > line_y and curr_y <= line_y):
                                # --- BLOK KALKULASI KECEPATAN DIMODIFIKASI ---
                                distance_pixels = math.sqrt((track[-1][0] - track[-2][0])**2 + (track[-1][1] - track[-2][1])**2)
                                # Asumsi kalibrasi: 10 piksel = 1 meter. Perlu disesuaikan.
                                pixel_to_meter = 0.1 
                                speed_kmh = (distance_pixels * pixel_to_meter * fps) * 3.6
                                
                                update_daily_stats(camera_id, cls, speed_kmh)
                                # --------------------------------------------
                                track_history.pop(track_id, None)
                
                time.sleep(0.02)

        except Exception as e:
            print(f"WORKER Exception di thread kamera {camera_id}: {e}")
            time.sleep(5)
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

def start_worker():
    """Memulai thread analisis untuk semua kamera di database."""
    print("WORKER: Memulai pemrosesan statistik untuk semua kamera...")
    try:
        for camera in cameras_collection.find({}):
            camera_id = camera.get("camera_id")
            if camera_id and camera_id not in processing_threads:
                thread = Thread(target=process_and_save_stats, args=(camera,), daemon=True)
                processing_threads[camera_id] = thread
                thread.start()
                print(f"WORKER: Thread analisis untuk kamera {camera_id} telah dimulai.")
    except Exception as e:
        print(f"WORKER: Gagal memulai thread analisis: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    start_worker()
    while True:
        print(f"WORKER: Analisis berjalan, thread aktif: {len(processing_threads)}")
        time.sleep(60)