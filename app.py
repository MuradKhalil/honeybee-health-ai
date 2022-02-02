import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from typing import Any, Dict
from fastapi import FastAPI


class Prediction(BaseModel):
  likely_class: str
  confidence: float

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
    return {'message': 'This is the homepage of the API'}

@app.post('/predict', response_model=Prediction)
async def prediction_route(request: Dict[Any, Any]):
    
    bee_image = read_img(request['file'])
    detection_boxes = request['detection_boxes']
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

        return {'label': likely_class, 'confidence': confidence}