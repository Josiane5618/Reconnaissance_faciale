title = Reconnaissance Faciale
package.name = reconnaissancefaciale
package.domain = org.example
source.dir = .
source.main = FaceRecognitionApp.py
version = 1.0

# Dépendances Python (versions stables)
requirements = 
    python3==3.12.10,
    kivy==2.1.0,
    numpy==1.23.5,
    opencv-python-headless==4.6.0.66,
    plyer==2.1.0,
    android
   dlib,
   sqlite3

# Permissions Android
android.permissions = 
    CAMERA,
    WRITE_EXTERNAL_STORAGE,
    READ_EXTERNAL_STORAGE,
    INTERNET

android.entrypoint = FaceRecognitionApp


# Architectures cibles
android.archs = armeabi-v7a, arm64-v8a

# Fichiers à inclure
source.include_exts = py,png,jpg,jpeg,kv,dat,db,xml,txt
assets.dir = assets
assets.include = **/*.dat,**/*.db,**/*.png,**/*.xml

# Configuration Android
orientation = portrait
fullscreen = 0
android.minapi = 21
android.maxapi = 33
android.sdk = 33
android.ndk_version = 25b
android.wakelock = True

# Chemins SDK/NDK (Linux/WSL)
android.sdk_path = /home/hapi/Android/Sdk
android.ndk_path = /home/hapi/Android/Sdk/ndk/25.1.8937393

# Icône
icon.filename = c:/Users/user/Reconnaissance_faciale/icon.png  # Définir le chemin de l'icône

# Options de build
[buildozer]
log_level = 2
warn_on_root = 1
target = android-33
