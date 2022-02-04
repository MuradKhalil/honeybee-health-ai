import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from fastapi import FastAPI, File, Form
from pydantic import BaseModel
from typing import List, Optional
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


def predict_health(model, bee_image):
    input_shape = model.layers[0].input_shape

    bee_image = keras.utils.array_to_img(bee_image)
    bee_image_resized = bee_image.resize((input_shape[1], input_shape[2]))
    numpy_image = np.array(bee_image_resized).reshape((input_shape[1], input_shape[2], input_shape[3]))
    prediction_array = np.array([numpy_image])
    
    predictions = model.predict(prediction_array)
    print(predictions)
    prediction = predictions[0]

    predicted_class = [np.argmax(prediction)]
    confidence_score = [np.amax(prediction)]

    return predicted_class, confidence_score



def crop_all_bees(detection_boxes, img):
    cropped_bees_array = []
    for bee_box in detection_boxes:
        cropped_bee_array = crop_bee(bee_box, img)
        cropped_bees_array.append(cropped_bee_array)
    return cropped_bees_array


def batch_predict_health(model, bees_array):
    input_shape = model.layers[0].input_shape
    bees_batch_array = []

    for bee in bees_array:
        bee_image = keras.utils.array_to_img(bee)
        bee_image_resized = bee_image.resize((input_shape[1], input_shape[2]))
        bee_image_arr = np.array(bee_image_resized).reshape((input_shape[1], input_shape[2], input_shape[3]))
        bee_image_arr = np.expand_dims(bee_image_arr, axis=0)
        bees_batch_array.append(bee_image_arr)

    bees_batch_array = np.vstack(np.array(bees_batch_array))

    predictions = model.predict(bees_batch_array)

    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.amax(predictions, axis=1)

    return predicted_classes, confidence_scores    

    



app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = keras.models.load_model('saved_model/bee_health_model_v2', custom_objects = {"f1_score": tfa.metrics.F1Score})




@app.get('/')
def index():
    return {'message': 'Bee health model is online.'}



@app.post('/predict', response_model=Prediction)
async def make_prediction(file: bytes = File(...), detection_boxes: Optional[str] = Form(None)):
    
    input_image = preprocess_image(file)

    # if no detection boxes input given, predict health on the entire image assuming that the input image is already a cropped bee image
    # this is only used when testing model performance directly with FastAPI model server, Swagger UI
    if detection_boxes == None:
        predicted_classes, confidence_scores = predict_health(model, input_image)
    

    # if detection box input provided, crop bees from image and then batch predict
    else:
        detection_boxes = json.loads(detection_boxes)

        cropped_bees_array = crop_all_bees(detection_boxes, input_image)
        predicted_classes, confidence_scores = batch_predict_health(model, cropped_bees_array)
        confidence_scores = confidence_scores.tolist()


    predicted_classes = pd.Series(predicted_classes) \
        .map(label_dict) \
        .tolist()

    return {'predictions': predicted_classes, 'confidence_scores': confidence_scores}