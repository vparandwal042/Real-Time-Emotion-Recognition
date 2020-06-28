from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

class FacialExpressionModel(object):

    EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        img_pixels = image.img_to_array(img)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        self.preds = self.loaded_model.predict(img_pixels)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
