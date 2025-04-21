import io
import os
import sqlite3
from tkinter import filedialog
import cv2 # type: ignore
import dlib # type: ignore
import numpy as np # type: ignore
from kivy.app import App # type: ignore
from kivy.clock import Clock # type: ignore
from kivy.core.image import Image as CoreImage # type: ignore
from kivy.graphics.texture import Texture # type: ignore
from kivy.uix.boxlayout import BoxLayout # type: ignore
from kivy.uix.button import Button # type: ignore
from kivy.uix.checkbox import CheckBox # type: ignore
from kivy.uix.gridlayout import GridLayout # type: ignore
from kivy.uix.image import Image # type: ignore
from kivy.uix.label import Label # type: ignore
from kivy.uix.popup import Popup # type: ignore
from kivy.uix.scrollview import ScrollView # type: ignore
from kivy.uix.textinput import TextInput # type: ignore

# Charger les modèles
predictor_path = os.path.join('assets', 'shape_predictor_68_face_landmarks.dat')
face_rec_model_path = os.path.join('assets', 'dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL,
    image BLOB
)
''')
conn.commit()

def add_or_update_face_in_db(image, name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        print("Pas de visage trouvé")
        return
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape)).astype(np.float32)
        face_descriptor_bytes = face_descriptor.tobytes()
        cursor.execute("SELECT * FROM faces WHERE encoding = ?", (face_descriptor_bytes,))
        if cursor.fetchone():
            print(f"Le visage pour {name} existe déjà.")
            return
        else:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_image = image[y:y+h, x:x+w]
            _, buffer = cv2.imencode('.png', face_image)
            face_image_binary = buffer.tobytes()
            cursor.execute("INSERT INTO faces (name, encoding, image) VALUES (?, ?, ?)", (name, face_descriptor_bytes, face_image_binary))
            conn.commit()
            print(f"Visage enregistré : {name}")

def recognize_face(image, seuil=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return None, None, None
    closest_name = None
    closest_image = None
    min_distance = float('inf')
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape)).astype(np.float32)
        cursor.execute("SELECT name, encoding, image FROM faces")
        for db_name, db_encoding, db_image in cursor.fetchall():
            db_encoding = np.frombuffer(db_encoding, dtype=np.float32)
            distance = np.linalg.norm(face_descriptor - db_encoding)
            if distance < min_distance:
                min_distance = distance
                closest_name = db_name
                closest_image = cv2.imdecode(np.frombuffer(db_image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if min_distance < seuil:
        return closest_name, closest_image, min_distance
    else:
        return None, None, None

class FaceRecognitionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.select_mode = False
        self.selected_faces = set()
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        self.show_main_screen()
        return self.layout

    def import_images_from_folder(self, folder_path='images'):
        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                name = os.path.splitext(filename)[0]
                image = cv2.imread(image_path)
                if image is not None:
                    add_or_update_face_in_db(image, name)
                    count += 1
        popup = Popup(title="Importation terminée ✅",
                      content=Label(text=f"{count} image(s) importée(s) depuis le dossier."),
                      size_hint=(0.6, 0.3))
        popup.open()

    def import_single_image(self, *args):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        root.destroy()
        if file_path:
            image = cv2.imread(file_path)
            name = os.path.splitext(os.path.basename(file_path))[0]
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detector(gray, 1)
                if len(faces) == 0:
                    popup = Popup(title="Aucun visage détecté ❌",
                                  content=Label(text="Aucun visage trouvé dans cette image."),
                                  size_hint=(0.6, 0.3))
                    popup.open()
                    return

                shape = predictor(gray, faces[0])
                face_descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape)).astype(np.float32)
                face_descriptor_bytes = face_descriptor.tobytes()

                cursor.execute("SELECT * FROM faces WHERE encoding = ?", (face_descriptor_bytes,))
                if cursor.fetchone():
                    popup = Popup(title="Doublon 🚫",
                                  content=Label(text="Ce visage est déjà enregistré dans la base."),
                                  size_hint=(0.6, 0.3))
                    popup.open()
                    return

                x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
                face_image = image[y:y+h, x:x+w]
                _, buffer = cv2.imencode('.png', face_image)
                face_image_binary = buffer.tobytes()
                cursor.execute("INSERT INTO faces (name, encoding, image) VALUES (?, ?, ?)", (name, face_descriptor_bytes, face_image_binary))
                conn.commit()

                preview_texture = Texture.create(size=(face_image.shape[1], face_image.shape[0]), colorfmt='bgr')
                preview_texture.blit_buffer(face_image.tobytes(), bufferfmt='ubyte', colorfmt='bgr')
                preview_texture.flip_vertical()

                popup_content = BoxLayout(orientation='vertical')
                popup_content.add_widget(Image(texture=preview_texture))
                popup_content.add_widget(Label(text=f"Image importée et enregistrée sous : {name}"))
                popup = Popup(title="Aperçu ✅", content=popup_content, size_hint=(0.8, 0.8))
                popup.open()

    def show_main_screen(self):
        self.layout.clear_widgets()
        self.info_label = Label(text="Détection de visage...", size_hint=(1, 0.1))
        self.layout.add_widget(self.info_label)
        self.image_widget = Image(size_hint=(1, 0.6))
        self.layout.add_widget(self.image_widget)
        self.name_input = TextInput(hint_text="Entrez votre nom", size_hint=(1, 0.1))
        self.layout.add_widget(self.name_input)
        self.register_button = Button(text="Enregistrer le visage", size_hint=(1, 0.1))
        self.register_button.bind(on_press=self.register_new_face)
        self.layout.add_widget(self.register_button)
        self.gallery_button = Button(text="Voir la galerie", size_hint=(1, 0.1))
        self.gallery_button.bind(on_press=self.show_gallery)
        self.layout.add_widget(self.gallery_button)
        self.import_button = Button(text="Importer les images du dossier", size_hint=(1, 0.1))
        self.import_button.bind(on_press=lambda x: self.import_images_from_folder())
        self.layout.add_widget(self.import_button)
        self.import_single_button = Button(text="Importer une image", size_hint=(1, 0.1))
        self.import_single_button.bind(on_press=self.import_single_image)
        self.layout.add_widget(self.import_single_button)
        self.quit_button = Button(text="Quitter", size_hint=(1, 0.1))
        self.quit_button.bind(on_press=self.stop)
        self.layout.add_widget(self.quit_button)

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            self.info_label.text = "Erreur de capture caméra"
            return
        closest_name, closest_image, min_distance = recognize_face(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            shape = predictor(gray, face)
            for i in range(68):
                point = shape.part(i)
                cv2.circle(frame, (point.x, point.y), 1, (0, 0, 255), -1)
        image_to_show = closest_image if closest_name and closest_image is not None else frame
        image_to_show = cv2.resize(image_to_show, (640, 480))
        image_rgb = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB)
        texture = Texture.create(size=(image_rgb.shape[1], image_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(image_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        self.image_widget.texture = texture
        self.info_label.text = f"Visage reconnu : {closest_name} - Distance : {min_distance:.2f}" if closest_name else "Visage inconnu"

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
        self.info_label.text = f"Visage enregistré : {name}"

    def show_gallery(self, instance=None):
        self.layout.clear_widgets()
        scroll = ScrollView()
        grid = GridLayout(cols=3, size_hint_y=None, spacing=10, padding=10)
        grid.bind(minimum_height=grid.setter('height'))
        cursor.execute("SELECT id, name, image FROM faces")
        for face_id, name, image_data in cursor.fetchall():
            if image_data:
                try:
                    buf = io.BytesIO(image_data)
                    img = CoreImage(buf, ext='png')
                    box = BoxLayout(orientation='vertical', size_hint_y=None, height=260)
                    image_widget = Image(texture=img.texture, size_hint_y=0.7)
                    image_widget.size_hint_x = 1
                    image_widget.size_hint_y = 1
                    image_widget.allow_stretch = True
                    image_widget.keep_ratio = False
                    box.add_widget(image_widget)
                    box.add_widget(Label(text=name, size_hint_y=0.15))
                    if self.select_mode:
                        checkbox = CheckBox(size_hint_y=0.15)
                        checkbox.bind(active=lambda cb, val, fid=face_id: self.on_select_toggle(fid, val))
                        box.add_widget(checkbox)
                    else:
                        btn = Button(text="🗑 Supprimer", size_hint_y=0.15)
                        btn.bind(on_press=lambda instance, fid=face_id: self.confirm_delete(fid))
                        box.add_widget(btn)
                    grid.add_widget(box)
                except Exception as e:
                    print(f"Erreur image : {e}")
        scroll.add_widget(grid)
        self.layout.add_widget(scroll)
        if self.select_mode:
            delete_button = Button(text="Supprimer la sélection", size_hint=(1, 0.1))
            delete_button.bind(on_press=self.delete_selected_faces)
            self.layout.add_widget(delete_button)
        toggle_button = Button(text="Retour" if self.select_mode else "Mode multi-sélection", size_hint=(1, 0.1))
        toggle_button.bind(on_press=lambda x: self.toggle_select_mode() if not self.select_mode else self.show_main_screen())
        self.layout.add_widget(toggle_button)
        if not self.select_mode:
            retour_btn = Button(text="Retour", size_hint=(1, 0.1))
            retour_btn.bind(on_press=lambda x: self.show_main_screen())
            self.layout.add_widget(retour_btn)

    def toggle_select_mode(self):
        self.select_mode = not self.select_mode
        self.selected_faces = set()
        self.show_gallery()

    def on_select_toggle(self, face_id, value):
        if value:
            self.selected_faces.add(face_id)
        else:
            self.selected_faces.discard(face_id)

    def delete_selected_faces(self, instance):
        for face_id in self.selected_faces:
            cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.commit()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces_temp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                image BLOB
            )
        ''')
        cursor.execute('INSERT INTO faces_temp (name, encoding, image) SELECT name, encoding, image FROM faces')
        cursor.execute('DROP TABLE faces')
        cursor.execute('ALTER TABLE faces_temp RENAME TO faces')
        conn.commit()
        self.select_mode = False
        self.selected_faces.clear()
        self.show_gallery()

    def confirm_delete(self, face_id):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        msg = Label(text="Voulez-vous vraiment supprimer ce visage ?", size_hint=(1, 0.5))
        btns = BoxLayout(size_hint=(1, 0.5), spacing=10)
        btn_yes = Button(text="Oui")
        btn_no = Button(text="Non")
        popup = Popup(title="Confirmation", content=content, size_hint=(0.8, 0.4))
        btn_yes.bind(on_press=lambda x: self.delete_face(face_id, popup))
        btn_no.bind(on_press=popup.dismiss)
        btns.add_widget(btn_yes)
        btns.add_widget(btn_no)
        content.add_widget(msg)
        content.add_widget(btns)
        popup.open()

    def delete_face(self, face_id, popup):
        cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.commit()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces_temp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                image BLOB
            )
        ''')
        cursor.execute('INSERT INTO faces_temp (name, encoding, image) SELECT name, encoding, image FROM faces')
        cursor.execute('DROP TABLE faces')
        cursor.execute('ALTER TABLE faces_temp RENAME TO faces')
        conn.commit()
        popup.dismiss()
        self.show_gallery()

    def on_stop(self):
        if self.capture.isOpened():
            self.capture.release()
        conn.close()

if __name__ == '__main__':
    FaceRecognitionApp().run()
