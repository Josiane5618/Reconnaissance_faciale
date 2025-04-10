[app]

# Configuration de base
title = Reconnaissance Faciale
package.name = reconnaissancefaciale
package.domain = org.example
source.dir = .
source.main = main.py
version = 1.0
requirements = python3,kivy,numpy,opencv-python,dlib,sqlite3

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

# Icone de l'application
icon.filename = assets/icon.png

# Android SDK Path
android.sdk_path = /Users/USERNAME/Library/Android/sdk

# Android NDK Path
android.ndk_path = /Users/USERNAME/Library/Android/sdk/ndk/21.3.6528147
