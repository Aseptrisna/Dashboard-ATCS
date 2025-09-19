import cv2
import time
import math
from collections import defaultdict
from threading import Thread, Lock
from ultralytics import YOLO
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

# Muat variabel lingkungan
load_dotenv()

# Kelas untuk menangani semua logika pemrosesan untuk satu stream kamera
class VideoProcessor:
    def __init__(self, camera_data):
        self.camera_id = camera_data.get("camera_id")
        self.rtsp_url = camera_data.get("rtsp_url")
        self.line_position = camera_data.get("calibration_line_position", 0.5)

        self.model = YOLO('yolov8n.pt')
        self.class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        # Atribut untuk data dan state
        self.latest_frame = None
        self.daily_stats = {
            "total_car": 0, "total_motorcycle": 0, "total_bus": 0, "total_truck": 0,
            "average_speed": 0
        }
        self.track_history = defaultdict(lambda: [])
        self.vehicle_speeds = defaultdict(float)
        
        self.is_running = False
        self.thread = None
        self.frame_lock = Lock() # Untuk sinkronisasi akses ke frame

        # Koneksi DB diinisialisasi per thread untuk keamanan thread
        self.mongo_client = None
        self.stats_collection = None
        self.db_uri = os.getenv("MONGO_URI")

    def _connect_db(self):
        """Membuat koneksi database."""
        try:
            self.mongo_client = MongoClient(self.db_uri, serverSelectionTimeoutMS=5000)
            db = self.mongo_client.atcs
            self.stats_collection = db.daily_stats
            print(f"[{self.camera_id}] Koneksi MongoDB berhasil.")
        except Exception as e:
            print(f"[{self.camera_id}] Gagal terhubung ke MongoDB: {e}")

    def start(self):
        """Memulai thread pemrosesan video."""
        if self.is_running:
            print(f"[{self.camera_id}] Prosesor sudah berjalan.")
            return
            
        self.is_running = True
        self.thread = Thread(target=self._process_stream, daemon=True)
        self.thread.start()
        print(f"[{self.camera_id}] Memulai pemrosesan untuk {self.rtsp_url}")

    def stop(self):
        """Menghentikan thread pemrosesan video."""
        self.is_running = False
        if self.thread:
            self.thread.join() # Tunggu thread selesai
        if self.mongo_client:
            self.mongo_client.close()
        print(f"[{self.camera_id}] Pemrosesan dihentikan.")

    def get_latest_frame(self):
        """Mengambil frame terbaru yang telah diproses dengan aman."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            
            ret, buffer = cv2.imencode('.jpg', self.latest_frame)
            return buffer.tobytes() if ret else None

    def get_stats(self):
        """Mengambil statistik terbaru."""
        return self.daily_stats

    def _update_daily_stats(self, vehicle_class, speed):
        """Memperbarui statistik di MongoDB."""
        if not self.stats_collection:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        class_name = self.class_names.get(vehicle_class)
        if not class_name:
            return

        query = {"camera_id": self.camera_id, "date": today}
        update = {
            "$inc": {f"total_{class_name}": 1},
            "$set": {"processed_at": datetime.now()}
        }
        
        # Inisialisasi data jika belum ada
        self.stats_collection.update_one(
            query,
            {"$setOnInsert": {"guid": f"VIDEO-{self.camera_id}-{today}"}},
            upsert=True
        )
        
        # Lakukan pembaruan
        self.stats_collection.update_one(query, update, upsert=True)

        # Perbarui statistik di memori untuk API
        self.daily_stats[f"total_{class_name}"] += 1

    def _process_stream(self):
        """Loop utama untuk memproses stream video."""
        self._connect_db() # Hubungkan ke DB di dalam thread
        
        while self.is_running:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                print(f"[{self.camera_id}] Gagal membuka stream. Mencoba lagi dalam 10 detik...")
                time.sleep(10)
                continue

            print(f"[{self.camera_id}] Berhasil terhubung ke stream.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30 # Default ke 30 jika fps tidak terbaca
            line_y = int(height * self.line_position)

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print(f"[{self.camera_id}] Frame kosong, stream mungkin terputus. Menghubungkan kembali...")
                    break 

                # Lakukan tracking dengan YOLOv8
                results = self.model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
                
                # Gambar garis virtual
                cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    classes = results[0].boxes.cls.int().cpu().tolist()
                    
                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        x, y, w, h = box
                        center = (int(x), int(y))
                        
                        track = self.track_history[track_id]
                        track.append(center)
                        if len(track) > 30:
                            track.pop(0)
                        
                        if len(track) > 1:
                            # Hitung kecepatan
                            pixel_to_meter = 0.1
                            distance_pixels = math.sqrt((track[-1][0] - track[-2][0])**2 + (track[-1][1] - track[-2][1])**2)
                            speed_kmh = (distance_pixels * pixel_to_meter * fps) * 3.6
                            self.vehicle_speeds[track_id] = speed_kmh

                            # Deteksi persilangan garis
                            prev_y = track[-2][1]
                            curr_y = track[-1][1]
                            
                            if (prev_y < line_y and curr_y >= line_y) or (prev_y > line_y and curr_y <= line_y):
                                self._update_daily_stats(cls, speed_kmh)
                                # Reset history agar tidak terhitung ganda
                                self.track_history[track_id] = []
                        
                        # Anotasi pada frame
                        class_name = self.class_names.get(cls, "other")
                        color = (0, 255, 0)
                        label = f"ID:{track_id} {class_name} {self.vehicle_speeds.get(track_id, 0):.1f}km/h"
                        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
                        cv2.putText(frame, label, (int(x - w / 2), int(y - h / 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                with self.frame_lock:
                    self.latest_frame = frame.copy()
            
            cap.release()
            time.sleep(2) # Beri jeda sebelum mencoba menghubungkan kembali
