docker stop $(docker ps -aqf "ancestor=proxy_server") # If there is currently running docker, stop it first
docker build --tag proxy_server . # Build docker image with Dockerfile
docker run -d --restart always -p 5000:5000 proxy_server