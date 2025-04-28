from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout

class FaceRecognitionApp(App):
    def build(self):
        # Créer un layout vertical
        layout = BoxLayout(orientation='vertical')
        
        # Ajouter un label
        label = Label(text="Application en cours d'exécution")
        
        # Ajouter un bouton pour quitter
        button = Button(text="Quitter")
        
        # Fonction de gestion du bouton pour quitter
        def on_button_press(instance):
            App.get_running_app().stop()
        
        button.bind(on_press=on_button_press)
        
        # Ajouter le label et le bouton au layout
        layout.add_widget(label)
        layout.add_widget(button)
        
        return layout

# Lancer l'application
if __name__ == '__main__':
    FaceRecognitionApp().run()
