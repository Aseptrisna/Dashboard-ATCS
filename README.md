Sistem Deteksi dan Analisis Kendaraan Berbasis Web
Proyek ini adalah versi web dari aplikasi deteksi kendaraan yang menggunakan model YOLOv8, Flask sebagai backend, dan MongoDB sebagai database. Sistem ini dirancang untuk memantau beberapa stream kamera CCTV (RTSP) secara bersamaan, melakukan analisis di latar belakang, dan menyajikan hasilnya melalui antarmuka web yang interaktif.

Fitur Utama
Daftar Kamera Dinamis: Halaman utama secara otomatis menampilkan semua kamera yang terdaftar di database MongoDB.

Streaming Real-time: Menampilkan video langsung dari kamera yang dipilih di halaman detail.

Analisis Latar Belakang: Proses deteksi dan tracking kendaraan berjalan secara terpisah untuk setiap kamera, memastikan antarmuka web tetap responsif.

Penyimpanan Data Analisis: Setiap kendaraan yang melintasi garis kalibrasi akan dicatat ke MongoDB, lengkap dengan timestamp, jenis kendaraan, dan ID kamera.

Dasbor Statistik: Halaman detail menampilkan ringkasan jumlah kendaraan yang terdeteksi pada hari ini, diperbarui secara real-time setiap beberapa detik.

Skalabilitas: Arsitektur multi-proses/multi-thread memungkinkan penambahan kamera baru tanpa perlu mengubah kode inti.

Arsitektur Sistem
Sistem ini terdiri dari dua komponen utama yang berjalan secara independen:

Aplikasi Web Flask (app.py): Bertanggung jawab untuk:

Menyajikan halaman HTML (daftar kamera dan halaman stream).

Menyediakan endpoint (/video_feed) yang men-stream frame video yang sudah diproses.

Menyediakan API (/api/stats) untuk data statistik yang akan dikonsumsi oleh JavaScript di frontend.

Prosesor Latar Belakang (background_processor.py): Bertanggung jawab untuk:

Membaca daftar kamera dari MongoDB.

Membuat thread terpisah untuk setiap kamera.

Setiap thread akan terhubung ke stream RTSP, menjalankan model YOLOv8 untuk deteksi, dan menghitung kendaraan.

Menyimpan hasil analisis ke koleksi vehicle_stats di MongoDB.

Menyimpan frame video yang sudah diolah ke dalam variabel bersama (shared_state) agar bisa diakses oleh aplikasi Flask.

Persyaratan
Python 3.8+

MongoDB Server

FFmpeg (seringkali dibutuhkan oleh OpenCV untuk menangani stream RTSP)

Cara Instalasi dan Menjalankan
1. Kloning Repositori
git clone <url-repo-anda>
cd <nama-folder-repo>

2. Instal Dependensi Python
Sangat disarankan untuk menggunakan virtual environment.

python -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
# venv\Scripts\activate  # Untuk Windows

pip install -r requirements.txt

3. Siapkan Database MongoDB
Pastikan server MongoDB Anda berjalan.

Buat database baru dengan nama traffic_data.

Buat dua koleksi (collection) di dalamnya: cameras dan vehicle_stats.

Masukkan data kamera Anda ke dalam koleksi cameras. Gunakan format berikut sebagai contoh:

{
  "camera_id": "1001",
  "rtsp_url": "rtsp://username:password@ip_address:port/stream_path",
  "location_name": "Simpang Dago",
  "latitude": -6.888,
  "longitude": 107.615,
  "calibration_line_position": 0.7
}

Penting: calibration_line_position adalah nilai antara 0.0 (atas) dan 1.0 (bawah) yang menentukan posisi garis virtual untuk penghitungan. Sesuaikan nilai ini untuk setiap kamera.

4. Jalankan Aplikasi
Anda perlu menjalankan dua proses di dua terminal yang berbeda.

Terminal 1: Jalankan Prosesor Latar Belakang
Proses ini harus dijalankan terlebih dahulu. Ia akan mulai menganalisis stream dan menyiapkan frame video.

python background_processor.py

Anda akan melihat log yang menandakan koneksi ke DB berhasil dan thread untuk setiap kamera dimulai.

Terminal 2: Jalankan Aplikasi Web Flask
Setelah prosesor latar belakang berjalan, jalankan server web Flask.

flask run --host=0.0.0.0 --port=5000

Opsi --host=0.0.0.0 membuat server dapat diakses dari perangkat lain di jaringan Anda.

5. Akses Aplikasi
Buka browser web Anda dan arahkan ke: http://localhost:5000

Anda akan melihat daftar kamera. Klik salah satu untuk melihat stream langsung dan statistiknya.