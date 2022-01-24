from fastapi import FastAPI, Request, File, UploadFile, Form
from PIL import Image
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import Optional
from io import BytesIO
import uvicorn
import numpy as np
import tensorflow as tf
import base64
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



# @app.get("/items/{id}", response_class=HTMLResponse)
# async def read_item(request: Request, id: str):
#     return templates.TemplateResponse("item.html", {"request": request, "id": id})


@app.post("/predict")
async def make_prediction(request: Request):
    form_data = await request.form()
    print(form_data)
    image = form_data['input_image']
    print(image)
    print(type(image))
    # contents = await input_image.read()

    # uploaded_file = form_data.get('input_image')
    # print(type(uploaded_file))
    #print (input_image)
    image = Image.open(BytesIO(image))
    print(image)
    # print(type(image))
    image.save('input_image.jpg')
    # input_image = base64.b64encode(input_image).decode('ascii')
    # print(input_image)

    

    # return templates.TemplateResponse("report.html", {"request": request, "input_image": input_image })
    return {'message': 'Honeybee health monitoring model is running!'}


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
