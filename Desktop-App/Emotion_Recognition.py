import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, QLabel, QVBoxLayout) 

from test2_ui import Ui_Form

path = "/home/vishal/Desktop/BEPROJECT/Real-Time-Emotion-Recognition/Models/"
#load model
model = model_from_json(open(path + "model_4layer_2_2_pool.json", "r").read())
#load weights
model.load_weights(path + 'model_4layer_2_2_pool.h5')

face_haar_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')

class video (QtWidgets.QDialog, Ui_Form):
    def __init__(self):
        super().__init__()                  

        self.setupUi(self)

        self.control_bt.clicked.connect(self.start_webcam)
        self.capture.clicked.connect(self.capture_image)
        self.capture.clicked.connect(self.startUIWindow)

        self.image_label.setScaledContents(True)

        self.cap = None

        self.timer = QtCore.QTimer(self, interval=5)
        self.timer.timeout.connect(self.update_frame)
        self._image_counter = 0

    @QtCore.pyqtSlot()
    def start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
             
        self.timer.start()

    @QtCore.pyqtSlot()
    def update_frame(self):
        ret, test_img = self.cap.read()
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            self.resized_img = cv2.resize(test_img, (640, 480))

        simage = cv2.flip(self.resized_img, 1)
        self.displayImage(self.resized_img, True)

    @QtCore.pyqtSlot()
    def capture_image(self):
        flag, frame = self.cap.read()
        path = r'/home/vishal/Desktop/BEPROJECT/Real-Time-Facial-Expression-Recognition/Images'
        if flag:
            QtWidgets.QApplication.beep()
            name = "my_image.jpg"
            cv2.imwrite(os.path.join(path, name), frame)
            self._image_counter += 1

    def displayImage(self, img, window=True):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))

    def startUIWindow(self):
        self.Window = UIWindow() 
        self.setWindowTitle("UIWindow")
        self.Window.ToolsBTN.clicked.connect(self.goWindow1)

        self.hide()
        self.Window.show()

    def goWindow1(self):
        self.show()
        self.Window.hide()

class UIWindow(QWidget):
    def __init__(self, parent=None):
        super(UIWindow, self).__init__(parent)

        self.resize(300, 300)
        self.label = QLabel("Check Image Folder", alignment=QtCore.Qt.AlignCenter)

        self.ToolsBTN = QPushButton('Back')

        self.v_box = QVBoxLayout()
        self.v_box.addWidget(self.label)
        self.v_box.addWidget(self.ToolsBTN)
        self.setLayout(self.v_box)


if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = video()
    window.setWindowTitle('Real-Time Emotion Recognition System')
    window.show()
    sys.exit(app.exec_())

