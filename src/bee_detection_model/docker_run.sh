docker stop $(docker ps -aqf "ancestor=bee_detection_model") # If there is currently running docker, stop it first
docker build --tag bee_detection_model . # Build docker image with Dockerfile
docker run -d --restart always -p 8000:8000 bee_detection_model