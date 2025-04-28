import io
import os
import sys
import sqlite3
import cv2
import dlib
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.checkbox import CheckBox
from kivy.utils import platform

if platform == 'android':
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.CAMERA, Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])

from plyer import filechooser

PREDICTOR_PATH = os.path.join('assets', 'shape_predictor_68_face_landmarks.dat')
REC_MODEL_PATH = os.path.join('assets', 'dlib_face_recognition_resnet_model_v1.dat')
DB_PATH = 'faces.db'

if not os.path.exists(PREDICTOR_PATH) or not os.path.exists(REC_MODEL_PATH):
    def show_error_popup(message):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        content.add_widget(Label(text=message))
        btn = Button(text="Fermer", size_hint=(1, None), height=40, background_normal='', background_color=(0.7,0,0,1), color=(1,1,1,1))
        content.add_widget(btn)
        popup = Popup(title="Erreur de configuration", content=content, size_hint=(0.9, 0.5), auto_dismiss=False)
        btn.bind(on_press=lambda *args: App.get_running_app().stop())
        popup.open()

    msg = (
        "\u274c Modèles manquants:\n"
        f"{PREDICTOR_PATH}\n{REC_MODEL_PATH}\n"
        "Téléchargez-les ici:\n"
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
        "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    )
    Clock.schedule_once(lambda dt: show_error_popup(msg), 0)

    class ErrorApp(App):
        def build(self):
            return Label(text="Vérification des modèles...")

    ErrorApp().run()
    sys.exit(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(REC_MODEL_PATH)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, encoding BLOB, image BLOB)''')
conn.commit()

def add_or_update_face_in_db(image, name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if not faces:
        return False
    face = faces[0]
    shape = predictor(gray, face)
    desc = np.array(face_rec_model.compute_face_descriptor(image, shape), dtype=np.float32)
    desc_bytes = desc.tobytes()
    cursor.execute("SELECT id FROM faces WHERE encoding=?", (desc_bytes,))
    if cursor.fetchone():
        return False
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    crop = image[y:y+h, x:x+w]
    _, buf = cv2.imencode('.png', crop)
    cursor.execute("INSERT INTO faces (name, encoding, image) VALUES (?,?,?)", (name, desc_bytes, buf.tobytes()))
    conn.commit()
    return True

def recognize_face(image, seuil=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    results = []
    for face in faces:
        shape = predictor(gray, face)
        desc = np.array(face_rec_model.compute_face_descriptor(image, shape), dtype=np.float32)
        cursor.execute("SELECT name, encoding FROM faces")
        best = (None, float('inf'))
        for db_name, db_enc in cursor.fetchall():
            db_vec = np.frombuffer(db_enc, dtype=np.float32)
            d = np.linalg.norm(desc - db_vec)
            if d < best[1]:
                best = (db_name, d)
        name, dist = best
        results.append((name, dist) if dist < seuil else (None, None))
    return results

class FaceRecognitionApp(App):
    def build(self):
        self.last_frame = None
        self.last_res = []
        root = BoxLayout(orientation='vertical')

        self.img_w = Image(size_hint=(1, 0.7))
        root.add_widget(self.img_w)

        self.info = Label(text="Aucun visage détecté", size_hint=(1, 0.05), font_size='14sp')
        root.add_widget(self.info)

        anchor = AnchorLayout(size_hint=(1, 0.25))
        btns = BoxLayout(size_hint=(None, None), height=55, spacing=8)
        colors = {
            'Enregistrer nom': (0.2,0.6,0.9,1),
            'Voir galerie': (0.2,0.6,0.9,1),
            'Importer image': (0.2,0.8,0.4,1),
            'Quitter': (0.9,0.3,0.3,1)
        }
        labels_callbacks = [
            ("Enregistrer nom", self.add_unknown),
            ("Voir galerie", self.show_gallery),
            ("Importer image", self.import_image),
            ("Quitter", lambda *_: self.stop()),
        ]
        for label, callback in labels_callbacks:
            btn = Button(
                text=label,
                size_hint=(None, None),
                size=(170, 45),
                background_normal='',
                background_color=colors[label],
                color=(1, 1, 1, 1),
                font_size='15sp'
            )
            btn.bind(on_press=callback)
            btns.add_widget(btn)
        btns.width = len(labels_callbacks) * (170 + 8)
        anchor.add_widget(btns)
        root.add_widget(anchor)

        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_frame, 1/30)
        return root

    def update_frame(self, dt):
        ret, frame = self.cap.read()
        if not ret: return
        self.last_frame = frame.copy()
        self.last_res = recognize_face(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        for (name, dist), face in zip(self.last_res, faces):
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            txt = f"{name}({dist:.2f})" if name else "Inconnu"
            cv2.putText(frame, txt, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tex = Texture.create(size=(img_rgb.shape[1], img_rgb.shape[0]), colorfmt='rgb')
        tex.blit_buffer(img_rgb.tobytes(), bufferfmt='ubyte', colorfmt='rgb')
        tex.flip_vertical()
        self.img_w.texture = tex
        texts = [f"{n} ({d:.2f})" if n else "Inconnu" for n,d in self.last_res]
        self.info.text = " | ".join(texts) if texts else "Aucun visage"

    def add_unknown(self, instance):
        if not any(n is None for n, _ in self.last_res): return
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        ti = TextInput(hint_text='Nom du visage', size_hint=(1, None), height=30)
        content.add_widget(ti)
        ok = Button(text='OK', size_hint=(1, None), height=40, background_normal='', background_color=(0.2,0.6,0.9,1), color=(1,1,1,1))
        content.add_widget(ok)
        popup = Popup(title='Enregistrer nom du visage', content=content, size_hint=(0.8,0.4), auto_dismiss=False)
        ok.bind(on_press=lambda *_: self._confirm_add(ti.text.strip(), popup))
        popup.open()

    def _confirm_add(self, name, popup):
        if name:
            add_or_update_face_in_db(self.last_frame, name)
            self.info.text = f"Nom '{name}' enregistré pour le visage"
        popup.dismiss()

    def import_image(self, instance=None):
        def callback(selection):
            if selection:
                path = selection[0]
                img = cv2.imread(path)
                if img is not None:
                    self.last_frame = img
                    self.last_res = recognize_face(img)
                    self.update_frame(0)
        filechooser.open_file(on_selection=callback)

    def show_gallery(self, instance):
        popup = Popup(title="Galerie des visages enregistrés", size_hint=(0.95, 0.95))
        scroll = ScrollView(size_hint=(1, 1))
        grid = GridLayout(cols=3, spacing=10, size_hint_y=None, padding=10)
        grid.bind(minimum_height=grid.setter('height'))
        checkboxes = {}
        selected_count_label = Label(text="0 sélectionné(s)", size_hint=(1, None), height=30)

        cursor.execute("SELECT name, image FROM faces")
        rows = cursor.fetchall()

        if not rows:
            grid.add_widget(Label(text="Aucun visage enregistré.", size_hint=(1, None), height=40))
        else:
            for name, img_data in rows:
                img_np = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img_np = cv2.resize(img_np, (150, 150))
                _, buf = cv2.imencode('.png', img_np)
                im = CoreImage(io.BytesIO(buf.tobytes()), ext='png')

                face_box = BoxLayout(orientation='vertical', size_hint_y=None, height=240, spacing=5, padding=5)
                img_layout = BoxLayout(orientation='vertical', size_hint=(None, 1), width=150, pos_hint={'center_x': 0.5})
                img_layout.add_widget(Image(texture=im.texture, size_hint=(1, None), height=150))
                face_box.add_widget(img_layout)

                name_label = Label(text=name, size_hint=(1, None), height=25, halign='center', pos_hint={'center_x': 0.5})
                face_box.add_widget(name_label)

                cb = CheckBox(size_hint=(None, None), size=(30, 30), pos_hint={'center_x': 0.5})
                checkboxes[name] = cb

                def update_count(cb_instance):
                    count = sum(1 for c in checkboxes.values() if c.active)
                    selected_count_label.text = f"{count} sélectionné(s)"

                cb.bind(active=lambda cb, val: update_count(cb))
                face_box.add_widget(cb)

                grid.add_widget(face_box)

        scroll.add_widget(grid)
        buttons = BoxLayout(size_hint=(1, None), height=50, spacing=10, padding=10)

        def delete_selected(*_):
            to_delete = [name for name, cb in checkboxes.items() if cb.active]
            for name in to_delete:
                cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
            if to_delete:
                conn.commit()
            popup.dismiss()
            self.show_gallery(None)

        def toggle_all(*_):
            all_active = all(cb.active for cb in checkboxes.values())
            for cb in checkboxes.values():
                cb.active = not all_active
            count = sum(1 for c in checkboxes.values() if c.active)
            selected_count_label.text = f"{count} sélectionné(s)"
            btn_toggle.text = "Tout désélectionner" if not all_active else "Tout sélectionner"

        btn_toggle = Button(text="Tout sélectionner", background_color=(0.6, 0.6, 0.6, 1), color=(1, 1, 1, 1))
        btn_toggle.bind(on_press=toggle_all)

        btn_suppr = Button(text="Supprimer la sélection", background_color=(0.9, 0.3, 0.3, 1), color=(1, 1, 1, 1))
        btn_suppr.bind(on_press=delete_selected)

        btn_close = Button(text="Retour", background_color=(0.2, 0.6, 0.9, 1), color=(1, 1, 1, 1))
        btn_close.bind(on_press=popup.dismiss)

        footer = BoxLayout(orientation='vertical', size_hint=(1, None), height=100, spacing=5, padding=5)
        footer.add_widget(selected_count_label)
        footer.add_widget(btn_toggle)
        buttons.add_widget(btn_suppr)
        buttons.add_widget(btn_close)
        footer.add_widget(buttons)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(scroll)
        layout.add_widget(footer)

        popup.content = layout
        popup.open()

if __name__ == "__main__":
    FaceRecognitionApp().run()
