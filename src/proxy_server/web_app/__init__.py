from flask import Flask
from flask_session import Session


app = Flask(__name__) # create app instance
sess = Session()
app.secret_key = 'gaeirogrioghogjfi'
app.config['SESSION_TYPE'] = 'filesystem'


import web_app.routes # import webapp routes