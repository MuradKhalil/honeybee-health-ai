from web_app import app
from flask import request, render_template, redirect, url_for, session
import requests, os
import uuid
import numpy as np
from web_app.functions import report_image
from dotenv import load_dotenv


# get environment variables
load_dotenv()
DETECTION_MODEL_URL = os.getenv('DETECTION_MODEL_URL')
HEALTH_MODEL_URL = os.getenv('HEALTH_MODEL_URL')
UPLOADS_DIR = os.getenv('UPLOADS_DIR')
RESULTS_DIR = os.getenv('RESULTS_DIR')
DEMO_FILEPATH = os.getenv('DEMO_FILEPATH')



@app.route('/')
def index():

    error_message = None
    # display error message if exists
    if session.get('error_message') != None:
        error_message = session.get('error_message')
        session.pop('error_message')

    return render_template("index.html", error_message = error_message)


@app.route('/about')
def about():
    return render_template("about.html")



@app.route("/predict", methods = ['POST', 'GET'])
def make_prediction():

    if request.method == "POST":
        # load and save input image file
        f = request.files['input_image']

        # if no file input, return error message.
        if f.filename == '':
            session['error_message'] = 'Please select an image file.'
            return redirect(url_for('index'))
            
        filename = f"{f.filename.split('.')[0]}_{uuid.uuid4().hex}.{f.filename.split('.')[-1]}" 
        full_filename = f"{UPLOADS_DIR}/{filename}" 
        f.save(full_filename)

    # if GET, show result with demo image
    else:
        full_filename = DEMO_FILEPATH

    # call bee detection model server to get bee detection box results
    detection_model_request = requests.post(DETECTION_MODEL_URL, files={'file': (full_filename, open(full_filename, 'rb'))})
    # os.remove(filename)
    obj_result = detection_model_request.json() # returns dictionary
    bees_count = len(obj_result['detection_scores'])
    

    # if bee detected, call bee health model server to get health predictions
    if bees_count > 0:
        # call bee health model server
        health_model_request = requests.post(HEALTH_MODEL_URL, files={'file': (full_filename, open(full_filename, 'rb'))}, data = {'detection_boxes': str(obj_result['detection_boxes'])})
        health_result = health_model_request.json()

        # create dummy health output json
        # health_labels = ["healthy", "varroa beetles", "ant problems", "hive being robbed", "missing queen"]
        # health_result = {
        #     'label': np.random.choice(health_labels, size = len(obj_result['detection_scores']), p=[0.8, 0.05, 0.05, 0.05, 0.05]).tolist(),
        #     'confidence': np.random.uniform(low=0.2, high=1, size=len(obj_result['detection_scores'])).tolist()
        # }

    else:
        health_result = {'predictions': [],
                        'confidence_scores': []}


    # create visualization report
    dest_fp = report_image(full_filename, obj_result, health_result, RESULTS_DIR)
    
    # count number of bees for each label
    counts = {
        'bees': bees_count,
        'healthy': health_result['predictions'].count("healthy"),
        'varroa beetles': health_result['predictions'].count("varroa beetles"),
        'ant problems': health_result['predictions'].count("ant problems"),
        'hive being robbed': health_result['predictions'].count("hive being robbed"),
        'missing queen': health_result['predictions'].count("missing queen"),
    }

    # print(obj_result)
    # print('')
    # print(health_result)
    # print('')
    # print(counts)

    return render_template("report.html", image_fp = "/".join(dest_fp.split('/')[1:]), result = health_result, counts = counts)



@app.route("/predict-demo")
def make_prediction_demo():
    return redirect(url_for('make_prediction'))
