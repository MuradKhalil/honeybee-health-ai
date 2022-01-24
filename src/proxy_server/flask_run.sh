python3 -m venv --clear venv
source venv/bin/activate
pip3 install --trusted-host pypi.python.org -r requirements.txt

export FLASK_APP=web_app
export FLASK_DEBUG=1
flask run --host=0.0.0.0 --port=5000 # For production, do not use flask run. Use a WSGI server (gunicorn, waitress, etc).  https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/