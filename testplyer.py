import cv2
from plyer import filechooser

from FaceRecognitionApp import recognize_face

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
