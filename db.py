# db.py

import os
from flask import current_app, g
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def get_db():
    """
    Membuka koneksi baru per permintaan dan menyimpannya di `g`.
    Jika koneksi sudah ada di `g`, koneksi tersebut akan digunakan kembali.
    """
    if 'db' not in g:
        try:
            # Mengambil URI dan nama DB dari konfigurasi aplikasi Flask
            mongo_uri = current_app.config['MONGO_URI']
            db_name = current_app.config['DB_NAME']
            
            # Membuat client dan menyimpannya di 'g' untuk ditutup nanti
            g.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            g.client.server_info() # Memaksa koneksi untuk memeriksa apakah server hidup
            g.db = g.client[db_name]
            print("Koneksi baru ke MongoDB dibuat.")
        except ConnectionFailure as e:
            print(f"FATAL: Gagal terhubung ke MongoDB. Error: {e}")
            # Mengembalikan None jika koneksi gagal agar bisa ditangani di route
            return None
            
    return g.db

def close_db(e=None):
    """
    Menutup koneksi database yang tersimpan di `g` saat konteks aplikasi ditutup.
    """
    client = g.pop('client', None)
    if client is not None:
        client.close()
        print("Koneksi MongoDB ditutup.")

def init_app(app):
    """
    Mendaftarkan fungsi close_db dengan aplikasi Flask agar dipanggil
    saat teardown.
    """
    app.teardown_appcontext(close_db)