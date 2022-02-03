import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, Form
from pydantic import BaseModel
from typing import List
from PIL import Image
from io import BytesIO
import json




label_dict = {
                0: "healthy",
                1: "varroa beetles",
                2: "ant problems",
                3: "hive being robbed",
                4: "missing queen"
            }


class Prediction(BaseModel):
    predictions: List[str] = []
    confidence_scores: List[float] = []


def preprocess_image(file, max_size=(1028, 1028)):
    image_raw = Image.open(BytesIO(file)).convert('RGB')
    image_raw.thumbnail(max_size, Image.ANTIALIAS) # rescale image to be smaller than max size
    # image_raw = tf.keras.preprocessing.image.img_to_array(image_raw)
    image_numpy = np.array(image_raw)
    img_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.uint8)
    
    return img_tensor


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
    model = keras.models.load_model('saved_model')



@app.get('/')
def index():
    return {'message': 'Bee health model is online.'}



@app.post('/predict', response_model=Prediction)
async def make_prediction(file: bytes = File(...), detection_boxes: str = Form(...)):
    
    beehive_image = preprocess_image(file)
    detection_boxes = json.loads(detection_boxes)
    print(detection_boxes)

    input_shape = model.layers[0].input_shape

    bees = []
    predicted_classes = []
    confidence_scores = []

    for bee_box in detection_boxes:
        cropped_bee_data = crop_bee(bee_box, beehive_image)
        
        cropped_bee_image = keras.utils.array_to_img(cropped_bee_data)
        cropped_bee_image_resized = cropped_bee_image.resize((input_shape[1], input_shape[2]))
        numpy_image = np.array(cropped_bee_image_resized).reshape((input_shape[1], input_shape[2], input_shape[3]))
        prediction_array = np.array([numpy_image])
        
        predictions = model.predict(prediction_array)
        prediction = predictions[0]

        predicted_class = np.argmax(prediction)
        confidence_score = np.amax(prediction)

        bees.append(cropped_bee_image)
        predicted_classes.append(predicted_class)
        confidence_scores.append(confidence_score)
    
    predicted_classes = pd.Series(predicted_classes) \
        .map(label_dict) \
        .tolist()

    return {'predictions': predicted_classes, 'confidence_scores': confidence_scores}