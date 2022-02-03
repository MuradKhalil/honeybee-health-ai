docker stop $(docker ps -aqf "ancestor=bee_health_model") # If there is currently running docker, stop it first
docker build --tag bee_health_model . # Build docker image with Dockerfile
docker run -d --restart always -p 7000:7000 bee_health_model