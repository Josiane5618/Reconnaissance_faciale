import dlib  # type: ignore
import cv2  # type: ignore
import numpy as np  # type: ignore
import sqlite3
from kivy.app import App  # type: ignore
from kivy.uix.boxlayout import BoxLayout  # type: ignore
from kivy.uix.image import Image  # type: ignore
from kivy.clock import Clock  # type: ignore
from kivy.graphics.texture import Texture  # type: ignore
from kivy.uix.label import Label  # type: ignore
from kivy.uix.textinput import TextInput  # type: ignore
from kivy.uix.button import Button  # type: ignore
from kivy.uix.filechooser import FileChooserIconView  # type: ignore
from kivy.uix.popup import Popup  # type: ignore
from kivy.uix.gridlayout import GridLayout  # type: ignore

# Charger les modèles
predictor_path = r"C:\Users\user\Desktop\Reconnaissance_faciale\assets\shape_predictor_68_face_landmarks.dat"
face_rec_model_path = r"C:\Users\user\Desktop\Reconnaissance_faciale\assets\dlib_face_recognition_resnet_model_v1.dat"

# Charger le détecteur de visage et les prédicteurs de points de repère
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Créer ou se connecter à la base de données SQLite
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

# Créer la table pour stocker les visages (si elle n'existe pas déjà)
cursor.execute(''' 
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL,
    image BLOB
) 
''')
conn.commit()

# Fonction pour ajouter ou mettre à jour un visage dans la base de données
def add_or_update_face_in_db(image, name):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        if len(faces) == 0:
            print("Pas de visage trouvé")
            return
        
        for face in faces:
            shape = predictor(gray, face)
            face_descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape))
            face_descriptor = face_descriptor.astype(np.float32)
            face_descriptor_bytes = face_descriptor.tobytes()

            cursor.execute("SELECT * FROM faces WHERE encoding = ?", (face_descriptor_bytes,))
            existing_face = cursor.fetchone()

            if existing_face:
                print(f"Le visage pour {name} existe déjà dans la base de données.")
                return  # Ignorer l'insertion pour les doublons
            else:
                print(f"Visage ajouté : {name}")
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                face_image = image[y:y+h, x:x+w]
                _, buffer = cv2.imencode('.png', face_image)
                face_image_binary = buffer.tobytes()

                cursor.execute(''' 
                INSERT INTO faces (name, encoding, image)
                VALUES (?, ?, ?)
                ''', (name, face_descriptor_bytes, face_image_binary))
                conn.commit()
                print(f"Visage enregistré : {name}")

    except Exception as e:
        print(f"Erreur lors de l'ajout ou mise à jour du visage dans la base de données: {e}")

# Fonction pour ajouter un visage à partir d'une image choisie par l'utilisateur
def add_face_from_file(image_path, name):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erreur lors de la lecture de l'image : {image_path}")
            return
        add_or_update_face_in_db(image, name)
    except Exception as e:
        print(f"Erreur lors de l'ajout de l'image : {e}")

# Fonction pour comparer un visage avec ceux de la base de données et retourner le nom, l'image et la distance
def recognize_face(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        
        if len(faces) == 0:
            return None, None, None  # Aucun visage trouvé, on retourne aussi None pour la distance
        
        closest_name = None
        closest_image = None
        min_distance = float('inf')  # Initialiser la distance à l'infini
        
        for face in faces:
            shape = predictor(gray, face)
            face_descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape))
            face_descriptor = face_descriptor.astype(np.float32)
            face_descriptor_bytes = face_descriptor.tobytes()
            
            cursor.execute("SELECT name, encoding, image FROM faces")
            faces_in_db = cursor.fetchall()

            for db_name, db_encoding, db_image in faces_in_db:
                db_encoding = np.frombuffer(db_encoding, dtype=np.float32)
                distance = np.linalg.norm(face_descriptor - db_encoding)  # Calcul de la distance entre les encodages
                
                if distance < min_distance:  # Si la distance est plus faible que la distance minimale
                    min_distance = distance
                    closest_name = db_name
                    
                    # Extraire l'image du visage correspondant à la base de données
                    closest_image = np.frombuffer(db_image, dtype=np.uint8)
                    closest_image = cv2.imdecode(closest_image, cv2.IMREAD_COLOR)

        return closest_name, closest_image, min_distance  # Retourner aussi la distance

    except Exception as e:
        print(f"Erreur dans la reconnaissance du visage : {e}")
        return None, None, None

# Fonction pour récupérer toutes les images de la base de données
def get_all_images_from_db():
    try:
        cursor.execute("SELECT image FROM faces")
        rows = cursor.fetchall()
        images = []
        
        for row in rows:
            image_binary = row[0]
            image = np.frombuffer(image_binary, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
        
        return images
    except Exception as e:
        print(f"Erreur lors de la récupération des images : {e}")
        return []

# Application Kivy pour la reconnaissance faciale en temps réel
class FaceRecognitionApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.info_label = Label(text="Détection de visage...", size_hint=(1, 0.1))
        layout.add_widget(self.info_label)

        self.image_widget = Image(size_hint=(1, 0.5))
        layout.add_widget(self.image_widget)

        self.name_input = TextInput(hint_text="Entrez votre nom", size_hint=(1, 0.1))
        layout.add_widget(self.name_input)

        self.register_button = Button(text="Enregistrer le visage", size_hint=(1, 0.1))
        self.register_button.bind(on_press=self.register_new_face)
        layout.add_widget(self.register_button)

        self.add_image_button = Button(text="Ajouter une image", size_hint=(1, 0.1))
        self.add_image_button.bind(on_press=self.open_filechooser)
        layout.add_widget(self.add_image_button)

        self.show_mosaic_button = Button(text="Afficher la Mosaïque", size_hint=(1, 0.1))
        self.show_mosaic_button.bind(on_press=self.show_mosaic)
        layout.add_widget(self.show_mosaic_button)

        self.quit_button = Button(text="Quitter", size_hint=(1, 0.1))
        self.quit_button.bind(on_press=self.stop)
        layout.add_widget(self.quit_button)

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Essayer avec DirectShow
        if not self.capture.isOpened():
            self.info_label.text = "Erreur caméra"
            return layout

        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.info_label.text = f"Caméra {int(width)}x{int(height)}"

        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # 30 FPS

        return layout

    def open_filechooser(self, instance):
        # Ouvre un filechooser pour choisir une image
        content = FileChooserIconView()
        content.bind(on_submit=self.on_file_selected)
        popup = Popup(title="Sélectionner une image", content=content, size_hint=(0.9, 0.9))
        popup.open()

    def on_file_selected(self, instance, selection, touch):
        if selection:
            image_path = selection[0]
            name = self.name_input.text.strip()
            if name:
                add_face_from_file(image_path, name)
                self.info_label.text = f"Image ajoutée pour {name}"
            else:
                self.info_label.text = "Veuillez entrer un nom avant d'ajouter une image."

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            self.info_label.text = "Erreur de capture caméra"
            return

        closest_name, closest_image, min_distance = recognize_face(frame)
        
        if closest_name:
            self.info_label.text = f"Visage reconnu : {closest_name} - Distance : {min_distance:.2f}"

            # Afficher l'image du visage le plus proche
            if closest_image is not None:
                # Redimensionner l'image du visage pour l'afficher
                closest_image = cv2.resize(closest_image, (300, 300))
                closest_image_rgb = cv2.cvtColor(closest_image, cv2.COLOR_BGR2RGB)
                
                texture = Texture.create(size=(closest_image_rgb.shape[1], closest_image_rgb.shape[0]), colorfmt='rgb')
                texture.blit_buffer(closest_image_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

                # Correction de l'orientation de la texture
                texture.flip_vertical()  # Flip vertical si nécessaire

                self.image_widget.texture = texture
            else:
                self.info_label.text = "Aucun visage correspondant trouvé"
        else:
            self.info_label.text = "Visage inconnu"

        # Traitement pour afficher le cadre vidéo en direct de la webcam
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Dessiner les points de repère sur le visage
            shape = predictor(gray, face)
            for i in range(68):  # shape contient 68 points
                x_point = shape.part(i).x
                y_point = shape.part(i).y
                cv2.circle(frame, (x_point, y_point), 1, (0, 0, 255), -1)  # Dessiner un petit cercle rouge pour chaque point

        # Conversion de l'image OpenCV (BGR) en RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Vérifier si l'image est mal orientée (plus haute que large)
        if frame_rgb.shape[0] > frame_rgb.shape[1]:
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)

        # Créer une texture Kivy à partir de l'image de la webcam
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

        # Correction de l'orientation de la texture si nécessaire
        texture.flip_vertical()  # Flip vertical pour corriger l'orientation

        # Affichage de la texture dans le widget Image
        self.image_widget.texture = texture

    def register_new_face(self, instance):
        ret, frame = self.capture.read()
        if not ret:
            self.info_label.text = "Erreur de capture caméra"
            return

        name = self.name_input.text.strip()
        if not name:
            self.info_label.text = "Veuillez entrer un nom"
            return

        add_or_update_face_in_db(frame, name)
        self.info_label.text = f"Visage enregistré ou mis à jour : {name}"

    def show_mosaic(self, instance):
        # Récupérer toutes les images de la base de données
        images = get_all_images_from_db()

        if not images:
            self.info_label.text = "Aucune image dans la base de données."
            return

        # Créer un GridLayout pour afficher les images
        mosaic_layout = GridLayout(cols=4, spacing=10, size_hint_y=None)
        mosaic_layout.bind(minimum_height=mosaic_layout.setter('height'))

        # Ajouter les images à la grille
        for image in images:
            # Convertir l'image en format RGB pour l'affichage dans Kivy
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Vérification et correction de l'orientation des visages
            if image_rgb.shape[0] > image_rgb.shape[1]:  # Si l'image est plus haute que large
                image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)  # Rotation de 90 degrés
            
            # Créer une texture à partir de l'image
            texture = Texture.create(size=(image_rgb.shape[1], image_rgb.shape[0]), colorfmt='rgb')
            texture.blit_buffer(image_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

            # Appliquer un flip vertical pour corriger l'orientation (si nécessaire)
            texture.flip_vertical()  # Flip vertical pour s'assurer que l'image est correctement orientée

            # Créer un widget Image avec la texture
            img_widget = Image(texture=texture, size_hint=(None, None), size=(150, 150))
            mosaic_layout.add_widget(img_widget)

        # Afficher la grille dans une popup
        mosaic_popup = Popup(title="Mosaïque d'images", content=mosaic_layout, size_hint=(0.9, 0.9))
        mosaic_popup.open()

    def on_stop(self):
        if self.capture.isOpened():
            self.capture.release()
        conn.close()

if __name__ == '__main__':
    FaceRecognitionApp().run()
