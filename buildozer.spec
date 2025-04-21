[app]
# Configuration de base
title = Reconnaissance Faciale
package.name = reconnaissancefaciale
package.domain = org.example
source.dir = .
source.main = face_recognition.py
version = 1.0
requirements = python3,kivy,numpy,opencv,sqlite3

# Permissions Android
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# Configuration matérielle
android.arch = armeabi-v7a, arm64-v8a

# Fichiers à inclure
source.include_exts = py,png,jpg,kv,dat,db
assets.dir = assets
assets.include = *.dat,*.db,*.png

# Orientation
orientation = portrait

# Version SDK
android.minapi = 21
android.ndk = 21b

# Android SDK Path (Vérifiez ce chemin sur votre machine)
android.sdk_path = C:/Users/user/AppData/Local/Android/Sdk

# Android NDK Path (Vérifiez ce chemin sur votre machine)
android.ndk_path = C:/Users/user/AppData/Local/Android/Sdk/ndk/21.3.6528147
