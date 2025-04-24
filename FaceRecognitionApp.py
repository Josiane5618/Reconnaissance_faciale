import io, os, sys, sqlite3, cv2, dlib, numpy as np
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
from kivy.uix.checkbox import CheckBox
from kivy.uix.anchorlayout import AnchorLayout
from kivy.utils import platform

# Permissions Android
if platform == 'android':
    from android.permissions import request_permissions, Permission
    request_permissions([
        Permission.CAMERA,
        Permission.READ_EXTERNAL_STORAGE,
        Permission.WRITE_EXTERNAL_STORAGE
    ])

# Plyer pour ouvrir une image (Android)
from plyer import filechooser

# Chemins des modèles
PREDICTOR_PATH = os.path.join('assets', 'shape_predictor_68_face_landmarks.dat')
REC_MODEL_PATH = os.path.join('assets', 'dlib_face_recognition_resnet_model_v1.dat')
DB_PATH = 'faces.db'

# Vérification des modèles
if not os.path.exists(PREDICTOR_PATH) or not os.path.exists(REC_MODEL_PATH):
    def show_error_popup(message):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        content.add_widget(Label(text=message))
        btn = Button(text="Fermer", size_hint=(1, None), height=40,
                     background_normal='', background_color=(0.7,0,0,1), color=(1,1,1,1))
        content.add_widget(btn)
        popup = Popup(title="Erreur de configuration", content=content,
                      size_hint=(0.9, 0.5), auto_dismiss=False)
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

# Initialisation
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
            'Ajouter nom': (0.2,0.6,0.9,1),
            'Voir galerie': (0.2,0.6,0.9,1),
            'Importer image': (0.2,0.8,0.4,1),
            'Quitter': (0.9,0.3,0.3,1)
        }
        labels_callbacks = [
            ("Ajouter nom", self.add_unknown),
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
        ok = Button(text='OK', size_hint=(1, None), height=40,
                    background_normal='', background_color=(0.2,0.6,0.9,1), color=(1,1,1,1))
        content.add_widget(ok)
        popup = Popup(title='Enregistrer visage inconnu', content=content,
                      size_hint=(0.8,0.4), auto_dismiss=False)
        ok.bind(on_press=lambda *_: self._confirm_add(ti.text.strip(), popup))
        popup.open()

    def _confirm_add(self, name, popup):
        if name:
            add_or_update_face_in_db(self.last_frame, name)
            self.info.text = f"Visage '{name}' ajouté"
        popup.dismiss()

    def import_image(self, instance=None):
        def callback(selection):
            if selection:
                path = selection[0]
                img = cv2.imread(path)
                if img is not None:
                    add_or_update_face_in_db(img, os.path.splitext(os.path.basename(path))[0])
                    
                    # Popup avec confirmation et bouton retour
                    popup = Popup(title='Image importée', size_hint=(0.6, 0.3))
                    
                    # Contenu du popup
                    content = BoxLayout(orientation='vertical', spacing=10, padding=10)
                    content.add_widget(Label(text='Import terminé !'))
                    
                    # Bouton retour
                    btn_ret = Button(
                        text='Retour',
                        size_hint=(1, None),
                        height=40,
                        background_normal='',
                        background_color=(0.6, 0.6, 0.6, 1),
                        color=(1, 1, 1, 1),
                        font_size='14sp'
                    )
                    content.add_widget(btn_ret)
                    
                    # Lier le bouton retour pour fermer le popup
                    btn_ret.bind(on_press=lambda *_: popup.dismiss())
                    
                    popup.content = content
                    popup.open()
        filechooser.open_file(on_selection=callback)

    def show_gallery(self, instance=None):
        popup = Popup(title='Galerie', size_hint=(0.9, 0.9))
        cont = BoxLayout(orientation='vertical')
        scroll = ScrollView(size_hint=(1, 0.85))
        grid = GridLayout(cols=3, spacing=10, size_hint_y=None, padding=10)
        grid.bind(minimum_height=grid.setter('height'))
        self.selected_ids = set()
        cursor.execute("SELECT id, image, name FROM faces")
        for fid, img_data, name in cursor.fetchall():
            buf = io.BytesIO(img_data)
            tex = CoreImage(buf, ext='png').texture
            box = BoxLayout(orientation='vertical', size_hint_y=None, height=200)
            img_widget = Image(texture=tex, size_hint=(1, None), height=120)  # Correction ici
            box.add_widget(img_widget)
            box.add_widget(Label(text=name, size_hint=(1, 0.15), font_size='12sp'))
            cb = CheckBox(size_hint=(1, 0.15))
            cb.bind(active=lambda cb, val, fid=fid: self.on_select(fid, val))
            box.add_widget(cb)
            grid.add_widget(box)
        scroll.add_widget(grid)
        cont.add_widget(scroll)

        bar = BoxLayout(size_hint=(1, 0.15), spacing=10, padding=10)
        btn_del = Button(text='Supprimer sélection', size_hint=(0.5, None), height=40,
                         background_normal='', background_color=(0.9, 0.3, 0.3, 1), color=(1, 1, 1, 1), font_size='14sp')
        btn_del.bind(on_press=self.delete_selected)
        btn_ret = Button(text='Retour', size_hint=(0.5, None), height=40,
                         background_normal='', background_color=(0.6, 0.6, 0.6, 1), color=(1, 1, 1, 1), font_size='14sp')
        btn_ret.bind(on_press=lambda *_: popup.dismiss())
        bar.add_widget(btn_del)
        bar.add_widget(btn_ret)
        cont.add_widget(bar)

        popup.content = cont
        popup.open()

    def on_select(self, fid, val):
        if val:
            self.selected_ids.add(fid)
        else:
            self.selected_ids.discard(fid)

    def delete_selected(self, instance=None):
        for fid in list(self.selected_ids):
            cursor.execute("DELETE FROM faces WHERE id=?", (fid,))
        conn.commit()
        self.show_gallery()

    def on_stop(self):
        if self.cap.isOpened():
            self.cap.release()
        conn.close()

if __name__ == '__main__':
    FaceRecognitionApp().run()
