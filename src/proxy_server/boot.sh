#!/bin/bash

# run server
gunicorn -b 0.0.0.0:5000 --workers 4 --threads 8 --timeout 0 web_app:app
## GCP Tips
# Increase the number of workers to be equal to CPU available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.