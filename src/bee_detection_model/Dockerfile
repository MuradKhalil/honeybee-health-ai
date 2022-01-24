FROM python:3.7

COPY . /bee_detection_model

ENV PYTHONPATH=/bee_detection_model
WORKDIR /bee_detection_model

RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["./boot.sh"]