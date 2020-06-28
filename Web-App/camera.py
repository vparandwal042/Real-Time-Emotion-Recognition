import cv2
from model import FacialExpressionModel
import numpy as np

path = "/home/vishal/Desktop/BEPROJECT/Real-Time-Emotion-Recognition/Models/"

facec = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
model = FacialExpressionModel(path + "model_4layer_2_2_pool.json", path + "model_4layer_2_2_pool.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),thickness=7)            
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi)

            cv2.putText(fr, pred, (int(x), int(y)), font, 1, (0,0,255), 2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
