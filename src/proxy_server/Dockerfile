FROM python:3.7

COPY . /proxy_server

ENV PYTHONPATH=/proxy_server
WORKDIR /proxy_server

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["./boot.sh"]