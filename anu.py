import cv2
import os
from flask import Flask, Response, render_template_string, url_for, jsonify
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from bson import json_util
import json

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Konfigurasi koneksi MongoDB dari .env ---
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DBNAME = os.getenv('MONGO_DBNAME', 'atcs_db')

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info() 
    db = client[MONGO_DBNAME]
    cameras_collection = db['cameras']
    results_collection = db['atcs_results']
    print(f"Koneksi ke MongoDB ({MONGO_DBNAME}) berhasil.")
except Exception as e:
    print(f"Error: Tidak bisa terhubung ke MongoDB. Pastikan MongoDB server berjalan dan variabel .env sudah benar. Error: {e}")
    cameras_collection = None
    results_collection = None

# Direktori tempat video disimpan
VIDEO_DIR = 'processed_videos'

# URL Logo Dishub Kota Bandung
DISHUB_LOGO_URL = "logo.png"

def generate_frames(filename):
    """Membaca video frame per frame dan yield sebagai respons multipart."""
    video_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(video_path) or not os.path.realpath(video_path).startswith(os.path.realpath(VIDEO_DIR)):
        print(f"Error: Percobaan akses file yang tidak valid: {filename}")
        return

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Tidak bisa membuka video di path: {video_path}")
        return

    while True:
        success, frame = video_capture.read()
        if not success:
            print(f"Video '{filename}' selesai. Mengulang dari awal.")
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    video_capture.release()

@app.route('/')
def index():
    """Halaman utama yang menampilkan daftar kamera dari MongoDB."""
    all_cameras = []
    if cameras_collection is not None:
        all_cameras = list(cameras_collection.find().sort("camera_id", 1))

    html_template = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard Monitoring ATCS</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
            body { 
                font-family: 'Poppins', sans-serif; 
                background-color: #eef1f5; 
                margin: 0;
                padding: 20px;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                min-height: 100vh;
            }
            .container { 
                width: 100%; 
                max-width: 900px; 
            }
            .header {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.07);
                margin-bottom: 30px;
                text-align: center;
            }
            .header img {
                height: 60px;
                margin-right: 20px;
            }
            .header h1 {
                color: #2c3e50;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 700;
            }
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
            }
            .card {
                background: #fff;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.07);
                overflow: hidden;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }
            .card a {
                text-decoration: none;
                color: inherit;
                display: block;
                padding: 25px;
            }
            .card-id {
                font-size: 1.5rem;
                font-weight: 700;
                color: #0056b3;
                margin-bottom: 8px;
            }
            .card-location {
                font-size: 1rem;
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="{{ logo_url }}" alt="Logo Dishub">
                <h1>Monitoring ATCS Kota Bandung</h1>
            </div>
            {% if cameras %}
                <div class="grid-container">
                    {% for camera in cameras %}
                        <div class="card">
                            <a href="{{ url_for('stream_page', camera_id=camera.camera_id) }}">
                                <div class="card-id">Kamera {{ camera.camera_id }}</div>
                                <div class="card-location">{{ camera.location_name }}</div>
                            </a>
                        </div>
                    {% endfor %}
                </div>
            {% else %}
                <p style="text-align:center;">Tidak ada kamera ditemukan atau koneksi ke MongoDB gagal.</p>
            {% endif %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, cameras=all_cameras, logo_url=DISHUB_LOGO_URL)

@app.route('/stream/<camera_id>')
def stream_page(camera_id):
    """Halaman yang menampilkan video stream dan data terbaru."""
    latest_result, camera_details = None, None
    if results_collection is not None and cameras_collection is not None:
        latest_result = results_collection.find_one({'camera_id': camera_id}, sort=[('processed_at', -1)])
        camera_details = cameras_collection.find_one({'camera_id': camera_id})
    
    if latest_result:
        latest_result['processed_time'] = datetime.fromtimestamp(latest_result['processed_at']).strftime('%d %B %Y, %H:%M:%S')

    html_template = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Streaming Kamera {{ camera_id }}</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
            body { font-family: 'Poppins', sans-serif; background-color: #eef1f5; margin: 0; padding: 20px; color: #333; }
            .top-bar { display: flex; align-items: center; justify-content: space-between; max-width: 1400px; margin: 0 auto 20px auto; }
            .header { display: flex; align-items: center; gap: 15px; }
            .header img { height: 50px; }
            .header h1 { font-size: 1.5rem; margin: 0; }
            .back-link a { text-decoration: none; background-color: #fff; color: #333; padding: 10px 15px; border-radius: 8px; font-weight: 600; box-shadow: 0 2px 5px rgba(0,0,0,0.1); transition: all 0.2s ease; }
            .back-link a:hover { background-color: #f0f0f0; }
            .main-container { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; max-width: 1400px; margin: auto; }
            .video-container, .stats-container { border-radius: 12px; background-color: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.07); padding: 20px; }
            .video-stream { width: 100%; height: auto; border-radius: 8px; background-color: #000; }
            h2 { border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 0; }
            .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px; }
            .stat-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
            .stat-card .value { font-size: 1.8rem; font-weight: 700; color: #0056b3; }
            .stat-card .label { font-size: 0.9rem; color: #666; }
            .timestamp { margin-top: 20px; text-align: center; font-size: 0.9rem; color: #888; }
            @media (max-width: 992px) { .main-container { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div class="top-bar">
            <div class="header">
                <img src="{{ logo_url }}" alt="Logo Dishub">
                <h1>Live Monitoring: {{ camera_details.location_name if camera_details else camera_id }}</h1>
            </div>
            <div class="back-link"><a href="{{ url_for('index') }}">&larr; Kembali</a></div>
        </div>
        
        {% if latest_result and camera_details %}
            <div class="main-container">
                <div class="video-container">
                    <h2>Live Stream</h2>
                    <img class="video-stream" src="{{ url_for('video_feed', filename=latest_result.filename_result) }}" alt="Video Stream">
                </div>
                <div class="stats-container">
                    <h2>Analitik Lalu Lintas</h2>
                    <div class="stats-grid">
                        <div class="stat-card"><div id="total_car" class="value">{{ latest_result.total_car }}</div><div class="label">Mobil</div></div>
                        <div class="stat-card"><div id="total_motorcycle" class="value">{{ latest_result.total_motorcycle }}</div><div class="label">Motor</div></div>
                        <div class="stat-card"><div id="total_bus" class="value">{{ latest_result.total_bus }}</div><div class="label">Bus</div></div>
                        <div class="stat-card"><div id="total_truck" class="value">{{ latest_result.total_truck }}</div><div class="label">Truk</div></div>
                    </div>
                    <div class="stats-grid" style="margin-top: 20px;">
                         <div class="stat-card" style="grid-column: 1 / -1;"><div id="average_speed" class="value">{{ "%.2f"|format(latest_result.average_speed) }}</div><div class="label">Kecepatan Rata-rata (km/jam)</div></div>
                    </div>
                    <div id="processed_time" class="timestamp">Diperbarui pada: {{ latest_result.processed_time }}</div>
                </div>
            </div>
        {% else %}
            <h1 style="text-align:center;">Data tidak ditemukan untuk kamera ID: {{ camera_id }}</h1>
        {% endif %}

        <script>
            setInterval(async () => {
                try {
                    const response = await fetch("{{ url_for('api_latest_result', camera_id=camera_id) }}");
                    if (!response.ok) throw new Error('Network response was not ok');
                    const data = await response.json();
                    if (data && Object.keys(data).length > 0) {
                        document.getElementById('total_car').textContent = data.total_car;
                        document.getElementById('total_motorcycle').textContent = data.total_motorcycle;
                        document.getElementById('total_bus').textContent = data.total_bus;
                        document.getElementById('total_truck').textContent = data.total_truck;
                        document.getElementById('average_speed').textContent = data.average_speed.toFixed(2);
                        document.getElementById('processed_time').textContent = 'Diperbarui pada: ' + data.processed_time;
                    }
                } catch (error) {
                    console.error('Gagal mengambil data terbaru:', error);
                }
            }, 10000); // 10000 milidetik = 10 detik
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, camera_id=camera_id, latest_result=latest_result, camera_details=camera_details, logo_url=DISHUB_LOGO_URL)

@app.route('/api/latest_result/<camera_id>')
def api_latest_result(camera_id):
    """API endpoint untuk mendapatkan data hasil terbaru dalam format JSON."""
    if results_collection is None:
        return jsonify({"error": "Koneksi database gagal"}), 500
        
    latest_result = results_collection.find_one({'camera_id': camera_id}, sort=[('processed_at', -1)])
    
    if not latest_result:
        return jsonify({})

    # Konversi ObjectId ke string dan format waktu
    latest_result['_id'] = str(latest_result['_id'])
    latest_result['processed_time'] = datetime.fromtimestamp(latest_result['processed_at']).strftime('%d %B %Y, %H:%M:%S')
    
    return jsonify(latest_result)

@app.route('/video_feed/<path:filename>')
def video_feed(filename):
    """Endpoint API yang menyediakan video stream."""
    return Response(generate_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

