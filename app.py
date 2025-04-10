import sqlite3
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.core.image import Image as CoreImage
from kivy.uix.label import Label

class MyApp(App):

    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Lire l'image depuis la base de données
        image_id = 1  # Remplacez par l'ID de l'image à afficher
        image_binary = self.get_image_from_db(image_id)

        if image_binary:
            try:
                # Convertir l'image binaire en un objet Image Kivy
                img = CoreImage(image_binary, ext='jpg')

                # Créer un widget Image avec l'image binaire
                kivy_image = Image(texture=img.texture)
                layout.add_widget(kivy_image)
            except Exception as e:
                layout.add_widget(Label(text=f"Erreur lors du chargement de l'image : {str(e)}"))
        else:
            # Si l'image n'existe pas
            layout.add_widget(Image(source='no_image.png'))  # Image par défaut si aucune image n'est trouvée
            layout.add_widget(Label(text="Aucune image trouvée dans la base de données"))

        return layout

    def get_image_from_db(self, image_id):
        # Connexion à la base de données SQLite
        try:
            conn = sqlite3.connect('images.db')
            cursor = conn.cursor()

            # Sélectionner l'image en binaire depuis la base de données
            cursor.execute('SELECT image FROM images WHERE id = ?', (image_id,))
            image_binary = cursor.fetchone()

            # Fermer la connexion
            conn.close()

            if image_binary:
                return image_binary[0]
            return None
        except sqlite3.Error as e:
            print(f"Erreur de base de données: {e}")
            return None

if __name__ == '__main__':
    MyApp().run()

