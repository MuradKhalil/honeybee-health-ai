from fastapi import FastAPI, File
import uvicorn
import tensorflow_hub as hub
from pydantic import BaseModel
from typing import List
from functions import run_detector, filter_bees, preprocess_and_save_input_image

# configs
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
# https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1 - faster but less accurate (min_score > 0.2)
# https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1 - slower but more accurate (min_score > 0.6)
filename = 'input_image.jpg'


class Prediction(BaseModel):
    detection_boxes: List[List[float]] = []
    detection_scores: List[float] = []



# start application
app = FastAPI()


@app.on_event("startup")
def load_model():
    global model
    model = hub.load(module_handle).signatures['default']


@app.get('/')
def index():
    return {'message': 'Bee detection model is online.'}


@app.post("/predict", response_model=Prediction)
async def make_prediction(file: bytes = File(...)):
    preprocess_and_save_input_image(file, filename)
    result = run_detector(model, filename)
    result_bees = filter_bees(result)

    return result_bees


if __name__ == "__main__":
    uvicorn.run("main:app")