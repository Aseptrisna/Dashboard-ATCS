# File ini digunakan untuk berbagi variabel antara proses yang berbeda (Flask app dan background processor).
# Ini adalah cara sederhana untuk IPC (Inter-Process Communication).

# Dictionary untuk menyimpan frame video terakhir dari setiap kamera
# Kunci: camera_id, Nilai: frame dalam format bytes (jpeg)
processed_frames = {}

# Dictionary untuk menyimpan lock threading untuk setiap frame
# Ini penting untuk mencegah race condition saat satu proses menulis frame
# dan proses lain membacanya secara bersamaan.
# Kunci: camera_id, Nilai: objek threading.Lock()
frame_locks = {}
