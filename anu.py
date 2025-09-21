import cv2
import os
from flask import Flask, Response, render_template_string, url_for, jsonify, send_from_directory
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Inisialisasi aplikasi Flask dengan path ke folder statis
app = Flask(__name__, static_folder='static')

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
VIDEO_DIR = 'Y:/atcs'

# Nama file logo di folder static
LOGO_FILENAME = "logo.png"

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
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            body { 
                font-family: 'Inter', sans-serif;
                background-color: #f0f4f8;
            }
        </style>
    </head>
    <body class="flex justify-center items-start min-h-screen p-4 sm:p-6 lg:p-8">
        <div class="w-full max-w-5xl">
            <header class="flex items-center justify-center p-6 bg-white rounded-xl shadow-md mb-8">
                <img src="{{ url_for('static', filename=logo_filename) }}" alt="Logo Dishub" class="h-16 mr-6">
                <h1 class="text-3xl font-bold text-gray-800 text-center">Monitoring ATCS Kota Bandung</h1>
            </header>
            
            {% if cameras %}
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                    {% for camera in cameras %}
                        <a href="{{ url_for('stream_page', camera_id=camera.camera_id) }}" class="group block">
                            <div class="bg-white p-6 rounded-xl shadow-md hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
                                <div class="flex items-center mb-3">
                                    <i class="fas fa-video text-2xl text-blue-500 mr-4"></i>
                                    <h2 class="text-2xl font-bold text-gray-700">Kamera {{ camera.camera_id }}</h2>
                                </div>
                                <p class="text-gray-600 flex items-center"><i class="fas fa-map-marker-alt text-lg text-gray-400 mr-3"></i>{{ camera.location_name }}</p>
                            </div>
                        </a>
                    {% endfor %}
                </div>
            {% else %}
                <div class="bg-white p-8 rounded-xl shadow-md text-center">
                    <i class="fas fa-exclamation-triangle text-4xl text-yellow-500 mb-4"></i>
                    <p class="text-lg text-gray-700">Tidak ada kamera yang ditemukan atau koneksi ke MongoDB gagal.</p>
                </div>
            {% endif %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, cameras=all_cameras, logo_filename=LOGO_FILENAME)

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
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            body { font-family: 'Inter', sans-serif; background-color: #f0f4f8; }
            .stat-card .value { transition: color 0.5s ease; }
            .stat-card.updating .value { color: #3b82f6; transform: scale(1.1); }
            #video-stream { transition: opacity 0.5s ease-in-out; }
        </style>
    </head>
    <body class="p-4 sm:p-6 lg:p-8">
        <div class="max-w-7xl mx-auto">
            <header class="flex flex-wrap items-center justify-between gap-4 mb-6">
                <div class="flex items-center gap-4">
                    <img src="{{ url_for('static', filename=logo_filename) }}" alt="Logo Dishub" class="h-12">
                    <div>
                        <h1 class="text-2xl font-bold text-gray-800">Live Monitoring</h1>
                        <p class="text-gray-600">{{ camera_details.location_name if camera_details else 'Kamera ' + camera_id }}</p>
                    </div>
                </div>
                 <div class="text-right">
                    <div id="clock" class="text-xl font-semibold text-gray-700"></div>
                    <div id="date" class="text-sm text-gray-500"></div>
                </div>
                <a href="{{ url_for('index') }}" class="bg-white text-gray-800 font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-gray-100 transition-colors duration-300 flex items-center"><i class="fas fa-arrow-left mr-2"></i>Kembali</a>
            </header>
            
            {% if latest_result and camera_details %}
                <main class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div class="lg:col-span-2 bg-black rounded-xl shadow-lg overflow-hidden">
                         <img id="video-stream" class="w-full h-full object-cover" src="{{ url_for('video_feed', filename=latest_result.filename_result) }}" alt="Video Stream">
                    </div>
                    <div class="bg-white rounded-xl shadow-lg p-6">
                        <h2 class="text-xl font-bold text-gray-800 border-b pb-3 mb-4">Analitik Lalu Lintas</h2>
                        <div class="grid grid-cols-2 gap-4">
                            <div id="card-car" class="stat-card bg-blue-50 p-4 rounded-lg text-center">
                                <i class="fas fa-car text-3xl text-blue-500 mb-2"></i>
                                <div id="total_car" class="value text-4xl font-bold text-gray-800">{{ latest_result.total_car }}</div>
                                <div class="label text-sm text-gray-600">Mobil</div>
                            </div>
                            <div id="card-motor" class="stat-card bg-green-50 p-4 rounded-lg text-center">
                                <i class="fas fa-motorcycle text-3xl text-green-500 mb-2"></i>
                                <div id="total_motorcycle" class="value text-4xl font-bold text-gray-800">{{ latest_result.total_motorcycle }}</div>
                                <div class="label text-sm text-gray-600">Motor</div>
                            </div>
                            <div id="card-bus" class="stat-card bg-yellow-50 p-4 rounded-lg text-center">
                                <i class="fas fa-bus text-3xl text-yellow-500 mb-2"></i>
                                <div id="total_bus" class="value text-4xl font-bold text-gray-800">{{ latest_result.total_bus }}</div>
                                <div class="label text-sm text-gray-600">Bus</div>
                            </div>
                            <div id="card-truck" class="stat-card bg-red-50 p-4 rounded-lg text-center">
                                <i class="fas fa-truck text-3xl text-red-500 mb-2"></i>
                                <div id="total_truck" class="value text-4xl font-bold text-gray-800">{{ latest_result.total_truck }}</div>
                                <div class="label text-sm text-gray-600">Truk</div>
                            </div>
                        </div>
                        <div id="card-speed" class="stat-card bg-gray-100 mt-4 p-4 rounded-lg text-center col-span-2">
                             <i class="fas fa-tachometer-alt text-3xl text-gray-600 mb-2"></i>
                             <div class="label text-sm text-gray-600">Kecepatan Rata-rata (km/jam)</div>
                             <div id="average_speed" class="value text-4xl font-bold text-gray-800">{{ "%.2f"|format(latest_result.average_speed) }}</div>
                        </div>
                        <div id="processed_time" class="text-center text-sm text-gray-500 mt-6">
                            <i class="fas fa-info-circle mr-1"></i>Diperbarui pada: {{ latest_result.processed_time }}
                        </div>
                    </div>
                </main>
            {% else %}
                <div class="bg-white p-8 rounded-xl shadow-md text-center mt-6">
                    <i class="fas fa-video-slash text-5xl text-red-500 mb-4"></i>
                    <h1 class="text-2xl font-bold text-gray-800">Data Tidak Ditemukan</h1>
                    <p class="text-gray-600 mt-2">Tidak ada data hasil proses yang ditemukan untuk kamera ID: {{ camera_id }}</p>
                </div>
            {% endif %}
        </div>
        <script>
            let currentFilename = "{{ latest_result.filename_result if latest_result else '' }}";

            function updateClock() {
                const clockElement = document.getElementById('clock');
                const dateElement = document.getElementById('date');
                if (clockElement && dateElement) {
                    const now = new Date();
                    const timeOptions = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
                    const dateOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
                    
                    clockElement.textContent = now.toLocaleTimeString('id-ID', timeOptions).replace(/\\./g, ':');
                    dateElement.textContent = now.toLocaleDateString('id-ID', dateOptions);
                }
            }
            setInterval(updateClock, 1000);
            updateClock();

            function updateStat(elementId, cardId, newValue) {
                const element = document.getElementById(elementId);
                if (element && element.textContent !== String(newValue)) {
                    element.textContent = newValue;
                    const card = document.getElementById(cardId);
                    if(card) {
                        card.classList.add('updating');
                        setTimeout(() => card.classList.remove('updating'), 1000);
                    }
                }
            }

            setInterval(async () => {
                try {
                    const response = await fetch("{{ url_for('api_latest_result', camera_id=camera_id) }}");
                    if (!response.ok) throw new Error('Network response was not ok');
                    const data = await response.json();
                    
                    if (data && Object.keys(data).length > 0) {
                        // Update stats with visual feedback
                        updateStat('total_car', 'card-car', data.total_car);
                        updateStat('total_motorcycle', 'card-motor', data.total_motorcycle);
                        updateStat('total_bus', 'card-bus', data.total_bus);
                        updateStat('total_truck', 'card-truck', data.total_truck);
                        updateStat('average_speed', 'card-speed', data.average_speed.toFixed(2));
                        
                        document.getElementById('processed_time').innerHTML = '<i class="fas fa-info-circle mr-1"></i>Diperbarui pada: ' + data.processed_time;

                        // Check for new video and switch source if necessary
                        if (data.filename_result && data.filename_result !== currentFilename) {
                            console.log(Beralih video dari ${currentFilename} ke ${data.filename_result});
                            currentFilename = data.filename_result;
                            const videoElement = document.getElementById('video-stream');
                            if (videoElement) {
                                videoElement.style.opacity = 0.5;
                                setTimeout(() => {
                                    videoElement.src = /video_feed/${data.filename_result};
                                    videoElement.onload = () => {
                                        videoElement.style.opacity = 1;
                                    };
                                }, 500);
                            }
                        }
                    }
                } catch (error) {
                    console.error('Gagal mengambil data terbaru:', error);
                }
            }, 10000); // 10 detik
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, camera_id=camera_id, latest_result=latest_result, camera_details=camera_details, logo_filename=LOGO_FILENAME)

@app.route('/api/latest_result/<camera_id>')
def api_latest_result(camera_id):
    """API endpoint untuk mendapatkan data hasil terbaru dalam format JSON."""
    if results_collection is None:
        return jsonify({"error": "Koneksi database gagal"}), 500
        
    latest_result = results_collection.find_one({'camera_id': camera_id}, sort=[('processed_at', -1)])
    
    if not latest_result:
        return jsonify({})

    latest_result['_id'] = str(latest_result['_id'])
    latest_result['processed_time'] = datetime.fromtimestamp(latest_result['processed_at']).strftime('%d %B %Y, %H:%M:%S')
    
    return jsonify(latest_result)

@app.route('/video_feed/<path:filename>')
def video_feed(filename):
    """Endpoint API yang menyediakan video stream."""
    return Response(generate_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=2345, debug=True)