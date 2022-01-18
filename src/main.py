from fastapi import FastAPI, Request, File, UploadFile
from PIL import Image
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import Optional
from io import BytesIO
import uvicorn
import numpy as np
import tensorflow as tf

# configs


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



@app.on_event("startup")
def load_model():
    global model
    # model = tf.keras.models.load_model(model_file)



@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("item.html", {"request": request, "id": id})


# @app.post("/predict")
# async def make_prediction(file: bytes = File(...)):

#     # pre-process input image
#     image_raw = Image.open(BytesIO(file)).convert('RGB').resize((32,32))
#     image_raw.save('input_image.jpg')
#     image = np.array(image_raw).astype('float32')
#     image = np.expand_dims(image, axis=0) # to match model input dimension
    
#     prediction_raw = model.predict(image)
#     predicted_idx = np.argmax(prediction_raw)
#     predicted_label = label_names[predicted_idx]
#     confidence = float(prediction_raw[:, predicted_idx][0])
#     return {"prediction": predicted_label, "confidence": confidence}


# @app.get('/test')
# def test():
#     return {'message': 'Honeybee health monitoring model is running!'}



if __name__ == "__main__":
    uvicorn.run("main:app")


## RESOURCES ## 
# FastAPI post image file: https://stackoverflow.com/a/62437063
# Image pre-processing with PIL: https://auth0.com/blog/image-processing-in-python-with-pillow/ 
