import numpy as np
from tensorflow import keras
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import uvicorn

class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: List[float] = []
  likely_class: int

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = keras.models.load_model('data/06_models/bee_health_model')

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API'}

@app.post('/prediction/', response_model=Prediction)
async def prediction_route(file: UploadFile=File(...)):
    
    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    # Read image contents
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # Resize image to expected input shape
    input_shape = model.layers[0].input_shape
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))

    # Convert image into numpy format
    numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

    # Generate prediction
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]
    likely_class = np.argmax(prediction)

    return {
      'filename': file.filename,
      'contenttype': file.content_type,
      'prediction': prediction.tolist(),
      'likely_class': likely_class
    }