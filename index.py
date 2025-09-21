# app.py

import os
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request, url_for, Response, g, current_app
import cv2
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
from math import ceil
from bson import ObjectId

# --- Konfigurasi dan Inisialisasi ---
load_dotenv()
app = Flask(__name__, template_folder='templates')

# --- KONFIGURASI PENTING ---
VIDEO_DIR = 'Z:/atcs' 
ITEMS_PER_PAGE = 5

# --- Konfigurasi Flask untuk Database ---
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
app.config["DB_NAME"] = os.getenv("DB_NAME", "atcs")

# --- Fungsi untuk Koneksi DB per-Request ---
def get_db():
    """
    Membuka koneksi baru per-permintaan dan menyimpannya di 'g'.
    Jika sudah ada, akan digunakan kembali dalam permintaan yang sama.
    """
    if 'db' not in g:
        try:
            g.client = MongoClient(current_app.config['MONGO_URI'], serverSelectionTimeoutMS=5000)
            g.client.server_info()  # Memaksa koneksi untuk memeriksa apakah server hidup
            g.db = g.client[current_app.config['DB_NAME']]
            print("Koneksi baru ke MongoDB dibuat.")
        except ConnectionFailure as e:
            print(f"FATAL: Gagal terhubung ke MongoDB. Error: {e}")
            return None
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    """
    Menutup koneksi database yang tersimpan di `g` secara otomatis
    setelah setiap permintaan selesai.
    """
    client = g.pop('client', None)
    if client is not None:
        client.close()
        print("Koneksi MongoDB ditutup.")

# --- Fungsi Utilitas ---
def generate_frames(filename, directory):
    """Membaca video dari path yang diberikan, frame per frame, dan yield sebagai respons multipart."""
    video_path = os.path.join(directory, filename)
    if not os.path.exists(video_path):
        print(f"Error: File video tidak ditemukan di path: {video_path}")
        return
    if not os.path.realpath(video_path).startswith(os.path.realpath(directory)):
        print(f"Peringatan Keamanan: Upaya mengakses path tidak valid dicegah: {video_path}")
        return
    
    while True:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print(f"Error: Tidak bisa membuka video di {video_path}. Mencoba lagi dalam 5 detik.")
            time.sleep(5)
            continue
        
        print(f"Streaming dimulai untuk: {filename}")
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        video_capture.release()
        print(f"Streaming selesai untuk {filename}, memulai ulang.")

# --- Rute Utama ---
@app.route('/')
def index():
    """Halaman utama yang menampilkan daftar kamera."""
    db = get_db()
    if db is None: 
        return "Error: Koneksi Database Gagal.", 500
    try:
        all_cameras = list(db.cameras.find({}))
        for cam in all_cameras:
            cam['_id'] = str(cam['_id'])
        return render_template('index.html', cameras=all_cameras)
    except Exception as e:
        return f"Error saat mengambil data kamera: {e}", 500

@app.route('/camera/<camera_id>')
def camera_detail(camera_id):
    """Memuat halaman detail kamera dengan data riwayat halaman pertama."""
    db = get_db()
    if db is None: 
        return "Error: Koneksi Database Gagal.", 500
    
    camera = db.cameras.find_one({"camera_id": camera_id})
    if not camera: 
        return "Kamera tidak ditemukan", 404
    camera['_id'] = str(camera['_id'])

    # Data Downloader
    downloader_history = list(db.downloader.find({'camera_id': camera_id}).sort('timestamp', DESCENDING).limit(ITEMS_PER_PAGE))
    for item in downloader_history:
        item['filename'] = os.path.basename(item.get('local_path') or item.get('filename', ''))
        item['_id'] = str(item['_id'])
    total_downloader = db.downloader.count_documents({'camera_id': camera_id})
    total_pages_downloader = ceil(total_downloader / ITEMS_PER_PAGE)

    # Data Hasil ATCS
    results_history = list(db.atcs_results.find({'camera_id': camera_id}).sort('processed_at', DESCENDING).limit(ITEMS_PER_PAGE))
    for item in results_history:
        item['_id'] = str(item['_id'])
    total_results = db.atcs_results.count_documents({'camera_id': camera_id})
    total_pages_results = ceil(total_results / ITEMS_PER_PAGE)

    return render_template(
        'detail.html', 
        camera=camera,
        downloader_history=downloader_history,
        total_pages_downloader=total_pages_downloader,
        results_history=results_history,
        total_pages_results=total_pages_results
    )

# --- API untuk Paginasi Riwayat ---
@app.route('/api/history/downloader/<camera_id>')
def api_history_downloader(camera_id):
    """API untuk mengambil data riwayat downloader secara paginasi."""
    db = get_db()
    if db is None: 
        return jsonify({"error": "Database connection failed"}), 500
    
    page = request.args.get('page', 1, type=int)
    skip = (page - 1) * ITEMS_PER_PAGE
    
    history_data = list(db.downloader.find({'camera_id': camera_id}).sort('timestamp', DESCENDING).skip(skip).limit(ITEMS_PER_PAGE))
    
    for item in history_data:
        item['_id'] = str(item['_id'])
        item['filename'] = os.path.basename(item.get('local_path') or item.get('filename', ''))

    return jsonify(history_data)

@app.route('/api/history/results/<camera_id>')
def api_history_results(camera_id):
    """API untuk mengambil data riwayat ATCS results secara paginasi."""
    db = get_db()
    if db is None: 
        return jsonify({"error": "Database connection failed"}), 500

    page = request.args.get('page', 1, type=int)
    skip = (page - 1) * ITEMS_PER_PAGE
    
    history_data = list(db.atcs_results.find({'camera_id': camera_id}).sort('processed_at', DESCENDING).skip(skip).limit(ITEMS_PER_PAGE))

    for item in history_data:
        item['_id'] = str(item['_id'])

    return jsonify(history_data)

# --- API dan Rute Lainnya ---
@app.route('/api/latest_videos/<camera_id>')
def api_latest_videos(camera_id):
    """API untuk mendapatkan path video mentah dan hasil olahan terbaru."""
    db = get_db()
    if db is None: 
        return jsonify({'error': 'Database connection failed'}), 500
    
    raw_video_url = None
    raw_doc = db.downloader.find_one({'camera_id': camera_id}, sort=[('timestamp', DESCENDING)])
    if raw_doc:
        filename = os.path.basename(raw_doc.get('local_path') or raw_doc.get('filename', ''))
        if filename: 
            raw_video_url = url_for('serve_raw_video', filename=filename)

    processed_video_url = None
    processed_doc = db.atcs_results.find_one({'camera_id': camera_id}, sort=[('processed_at', DESCENDING)])
    if processed_doc and processed_doc.get('filename_result'):
        processed_video_url = url_for('serve_processed_video', filename=processed_doc['filename_result'])
        
    return jsonify({'raw_video_url': raw_video_url, 'processed_video_url': processed_video_url})

@app.route('/stats/<camera_id>')
def get_stats(camera_id):
    """API untuk statistik harian."""
    db = get_db()
    if db is None: 
        return jsonify({'error': 'Database connection failed'}), 500
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    stats_data = db.stats_collection.find_one({"camera_id": camera_id, "date": today_str})
    
    if stats_data:
        stats_data['_id'] = str(stats_data['_id'])
        return jsonify(stats_data)
    else:
        return jsonify({
            "total_car": 0, 
            "total_motorcycle": 0, 
            "total_bus": 0, 
            "total_truck": 0, 
            "average_speed": 0.0
        })

# --- Rute untuk Streaming Video ---
@app.route('/videos/raw/<path:filename>')
def serve_raw_video(filename):
    """Menyajikan video RAW sebagai stream MJPEG."""
    return Response(generate_frames(filename, VIDEO_DIR), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videos/processed/<path:filename>')
def serve_processed_video(filename):
    """Menyajikan video olahan sebagai stream MJPEG."""
    return Response(generate_frames(filename, VIDEO_DIR), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Blok Eksekusi Utama ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5079, debug=True)