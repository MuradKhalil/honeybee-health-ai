FROM python:3.8

COPY . /bee_health_model

ENV PYTHONPATH=/bee_health_model
WORKDIR /bee_health_model

RUN pip install -r requirements.txt

EXPOSE 7000

CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=7000"]