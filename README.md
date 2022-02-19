# Bee Health Monitoring

## Overview

Honeybee Health AI is an ML-based web application for beekeepers that predicts health of the bees from beehive images. We used a pre-trained object detection model to detect bees and trained a convolutional neural network model to predict the health of each bee. We built our ML product using microservices architecture and deployed our containerized services to GCP

The model for predicting bee health has been trained on 5000 single bee images with 5 target values: `healthy`, `varroa mites`, `ant problems`, `hive being robbed`, and `missing queen`. The data comes from the Kaggle platform and is in the form of images that have been cropped from a video file. The model achieves high predictive performance with metrics like acccuracy and f1 score being 0.97 and 0.90 respectiverly. While the performance is quite high, in this project it is important to control for false negative - reporting healthy bee while it is actually being infected.