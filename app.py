import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File
from pydantic import BaseModel
from typing import List
from PIL import Image
from io import BytesIO


class Prediction(BaseModel):
  label: List[str] = []
  confidence: List[float] = []


def preprocess_and_save_input_image(img, input_filename, max_size=(1028, 1028)):
    image_raw = Image.open(BytesIO(img)).convert('RGB')
    image_raw.thumbnail(max_size, Image.ANTIALIAS) # rescale image to be smaller than max size
    image_raw.save(input_filename)



def read_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def crop_bee(detection_box, img):    
    
    img_height, img_width = img.shape[:2]
    ymin, xmin, ymax, xmax = detection_box

    x_up = int(xmin*img_width)
    y_up = int(ymin*img_height)
    x_down = int(xmax*img_width)
    y_down = int(ymax*img_height)
    
    img = img[y_up:y_down, x_up:x_down, :]
    
    return img



app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = keras.models.load_model('data/06_models/bee_health_model')

@app.get('/')
def index():
    return {'message': 'Bee health model is online.'}

@app.post('/predict', response_model=Prediction)
async def prediction_route(file: bytes = File(...), detection_boxes: List[List[float]] = []):
    
    bee_image = read_img(file)

    # print(bee_image)

    # return {'label': [], 'confidence': []}


    input_shape = model.layers[0].input_shape

    bees = []
    classes = []
    confidences = []

    for bee_box in detection_boxes:
        cropped_bee_data = crop_bee(bee_box, bee_image)
        cropped_bee_image = keras.utils.array_to_img(cropped_bee_data)
        cropped_bee_image_resized = cropped_bee_image.resize((input_shape[1], input_shape[2]))
        numpy_image = np.array(cropped_bee_image_resized).reshape((input_shape[1], input_shape[2], input_shape[3]))
        prediction_array = np.array([numpy_image])
        predictions = model.predict(prediction_array)
        prediction = predictions[0]
        likely_class = np.argmax(prediction)
        confidence = np.amax(prediction)

        bees.append(cropped_bee_image)
        classes.append(likely_class)
        confidences.append(confidence)

        mydict = {
            0: "healthy",
            1: "varroa beetles",
            2: "ant problems",
            3: "hive being robbed",
            4: "missing queen"
        }
        
        classes = pd.Series(classes) \
            .map(mydict) \
            .tolist()

        return {'label': classes, 'confidence': confidences}