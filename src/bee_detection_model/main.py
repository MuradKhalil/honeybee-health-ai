from fastapi import FastAPI, Request, File, UploadFile
from PIL import Image
from tensorflow.python.lib.io import file_io
from typing import Optional
from io import BytesIO
import uvicorn
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from functions import run_detector, filter_bees


# model_file = file_io.FileIO('gs://honey-bees/faster_rcnn_openimages_v4_inception_resnet_v2_1/saved_model.pb', mode='rb')
# temp_model_path = './bee_detection_model.h5'
# temp_model_file = open(temp_model_path, 'wb')
# temp_model_file.write(model_file.read())
# temp_model_file.close()
# model_file.close()



app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    # https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1 - faster but less accurate (min_score > 0.2)
    # https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1 - slower but more accurate (min_score > 0.6)
    model = hub.load(module_handle).signatures['default']



@app.get('/')
def index():
    return {'message': 'Honeybee detection model is online.'}


@app.post("/predict")
async def make_prediction(file: bytes = File(...)):
    
    maxsize = (1028, 1028)
    filename = 'input_image.jpg'


    # pre-process input image
    image_raw = Image.open(BytesIO(file)).convert('RGB')
    image_raw.thumbnail(maxsize, Image.ANTIALIAS) # rescale image to be smaller than max size
    image_raw.save(filename)
    # image = np.array(image_raw).astype('float32')
    # image = np.expand_dims(image, axis=0) # to match model input dimension
    result = run_detector(model, filename)
    result_bees = filter_bees(result)
    return result_bees

if __name__ == "__main__":
    uvicorn.run("main:app")


## RESOURCES ## 
# FastAPI post image file: https://stackoverflow.com/a/62437063
# Image pre-processing with PIL: https://auth0.com/blog/image-processing-in-python-with-pillow/ 
