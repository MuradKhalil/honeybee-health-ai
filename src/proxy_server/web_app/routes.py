from web_app import app
from flask import request, render_template, url_for, redirect, json, jsonify
import requests, os
import uuid

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def make_prediction():

    # get input image file
    f = request.files['input_image']
    filename = f"{f.filename.split('.')[0]}_{uuid.uuid4().hex}.{f.filename.split('.')[-1]}" 
    full_filename = f"web_app/static/img/{filename}" 
    f.save(full_filename)

    # call bee detection model server
    # os.environ['NO_PROXY'] = '192.168.1.10'
    r = requests.post('http://192.168.1.10:8000/predict', files={'file': (full_filename, open(full_filename, 'rb'))})
    # os.remove(filename)
    result = r.json() # dictionary
    print(result)
    # call bee health model server
    # input: {'file': hive_image, 'result': filtered_bee}
    # output: {'label': ['healthy', 'varroa beetle', ...], 'confidence': [0.3456, 0.5352, ...]}
    return render_template("report.html", input_image = filename)
