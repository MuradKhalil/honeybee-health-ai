#!/bin/bash

# install packages
apt-get update \
    && apt-get install -y \
        git vim
# run server
gunicorn -b 0.0.0.0:5000 --workers=8 web_app:app