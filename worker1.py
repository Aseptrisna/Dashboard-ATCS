import os
import cv2
import time
import uuid
from datetime import datetime
from collections import defaultdict
from threading import Thread
from pymongo import MongoClient
from dotenv import load_dotenv
from ultralytics import YOLO

# --- Konfigurasi dan Inisialisasi ---
print("🚀 Worker Analisis CCTV Dimulai...")

# 1. Muat variabel lingkungan dari file .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# 2. Inisialisasi koneksi ke MongoDB
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # Cek koneksi
    db = client.get_database() # Mengambil nama database dari URI
    cameras_collection = db["cameras"]
    stats_collection = db["stats_collection"]
    print("✅ Berhasil terhubung ke MongoDB.")
except Exception as e:
    print(f"❌ Gagal terhubung ke MongoDB: {e}")
    exit() # Keluar dari skrip jika tidak bisa terhubung ke DB

# 3. Muat Model YOLOv8 (hanya sekali saat startup)
print("🧠 Memuat model YOLOv8...")
model = YOLO('yolov8n.pt')
print("✅ Model berhasil dimuat.")

# 4. Mapping class ID ke nama yang lebih sederhana untuk field di DB
CLASS_NAMES_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# --- Fungsi Inti Worker ---

# Anda perlu CLASS_NAMES_MAP dan stats_collection didefinisikan di luar fungsi ini
# CLASS_NAMES_MAP = { 2: "car", 3: "motorcycle", 5: "bus", 7: "truck" }
# stats_collection = db["stats_collection"]

def update_daily_stats(camera_id, class_id):
    """
    Memperbarui statistik harian di MongoDB (Create jika belum ada, Update jika sudah ada).
    Versi ini sudah memperbaiki masalah konflik update.
    """
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        vehicle_type_key = CLASS_NAMES_MAP.get(class_id)
        if not vehicle_type_key:
            return

        # Query untuk mencari dokumen berdasarkan camera_id dan tanggal hari ini
        query = {
            "camera_id": camera_id,
            "date": today_str
        }

        # --- BAGIAN KUNCI YANG DIPERBAIKI ---
        
        # 1. Siapkan semua field counter yang mungkin diinisialisasi
        initial_counts = {
            "total_car": 0,
            "total_motorcycle": 0,
            "total_bus": 0,
            "total_truck": 0
        }

        # 2. Tentukan field mana yang akan di-increment pada panggilan ini
        field_to_increment = f"total_{vehicle_type_key}"

        # 3. Hapus field yang akan di-increment dari daftar inisialisasi.
        #    Ini adalah langkah krusial untuk menghindari konflik.
        #    Jika dokumen baru, $inc akan otomatis membuat field ini dengan nilai 1.
        if field_to_increment in initial_counts:
            del initial_counts[field_to_increment]
        
        # 4. Buat perintah update yang aman dari konflik
        update_command = {
            '$inc': {
                field_to_increment: 1
            },
            '$setOnInsert': {
                'guid': str(uuid.uuid4()),
                'camera_id': camera_id,
                'date': today_str,
                'average_speed': 0.0,
                'created_at': datetime.now(),
                **initial_counts  # Tambahkan sisa counter yang tidak di-increment
            }
        }

        # Jalankan perintah dengan upsert=True
        stats_collection.update_one(query, update_command, upsert=True)

    except Exception as e:
        print(f"[{camera_id}] ❌ Error saat memperbarui statistik: {e}")



def process_camera_stream(camera_data):
    """
    Fungsi utama yang berjalan di thread untuk setiap kamera.
    - Membaca stream RTSP.
    - Melakukan deteksi dan tracking kendaraan.
    - Menghitung kendaraan yang melintasi garis virtual.
    - Memanggil fungsi untuk menyimpan hasil ke MongoDB.
    """
    camera_id = camera_data["camera_id"]
    rtsp_url = camera_data["rtsp_url"]
    line_position_ratio = camera_data.get("calibration_line_position", 0.5)
    
    track_history = defaultdict(lambda: [])
    
    while True:
        cap = None  # Inisialisasi cap
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"[{camera_id}] ❌ Error: Gagal membuka stream. Mencoba lagi dalam 15 detik...")
                time.sleep(15)
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            line_y = int(height * line_position_ratio)
            
            print(f"[{camera_id}] ▶️ Memulai pemrosesan untuk stream di lokasi: {camera_data.get('location_name', 'N/A')}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"[{camera_id}] ⚠️ Stream terputus. Mencoba menghubungkan kembali...")
                    break # Keluar dari loop dalam untuk re-inisialisasi VideoCapture

                # Lakukan tracking dengan YOLOv8, hanya untuk kelas yang relevan
                results = model.track(frame, persist=True, classes=list(CLASS_NAMES_MAP.keys()), verbose=False)
                
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    classes = results[0].boxes.cls.int().cpu().tolist()
                    
                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        _, y, _, _ = box # Kita hanya butuh posisi y untuk deteksi garis
                        
                        # Simpan histori posisi y untuk deteksi persilangan garis
                        track = track_history[track_id]
                        track.append(int(y))
                        if len(track) > 2:
                            track.pop(0)

                        # Cek jika kendaraan melintasi garis (dari atas ke bawah atau sebaliknya)
                        if len(track) == 2 and (track[0] < line_y and track[1] >= line_y):
                            # Uncomment baris di bawah untuk melihat log deteksi di konsol
                            # print(f"[{camera_id}] DETECTED: {CLASS_NAMES_MAP.get(cls)} (ID: {track_id}) melintasi garis.")
                            update_daily_stats(camera_id, cls)
                            # Hapus history agar kendaraan tidak dihitung berulang kali
                            track_history.pop(track_id, None)
                
        except Exception as e:
            print(f"[{camera_id}] 💥 Terjadi exception pada thread: {e}")
            time.sleep(10) # Beri jeda sebelum mencoba lagi
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            print(f"[{camera_id}] Melepas sumber video. Akan mencoba lagi.")
            time.sleep(5) # Jeda singkat sebelum loop utama mencoba lagi

def main():
    """
    Fungsi utama untuk memulai dan mengelola thread worker.
    """
    # ID Kamera yang akan dianalisis (sesuai permintaan)
    target_camera_ids = ["1001"] # <-- GANTI ATAU TAMBAHKAN ID KAMERA DI SINI
    
    print(f"🎯 Mencari data untuk kamera dengan ID: {target_camera_ids}")

    threads = []
    for cam_id in target_camera_ids:
        camera_document = cameras_collection.find_one({"camera_id": cam_id})
        
        if camera_document:
            print(f"✅ Data untuk kamera {cam_id} ditemukan. Memulai thread analisis...")
            # Membuat dan memulai thread untuk setiap kamera
            thread = Thread(target=process_camera_stream, args=(camera_document,), daemon=True)
            thread.start()
            threads.append(thread)
        else:
            print(f"⚠️ Peringatan: Data untuk kamera dengan ID '{cam_id}' tidak ditemukan di koleksi 'cameras'.")
            
    # Jaga agar skrip utama tetap berjalan selama thread aktif
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()