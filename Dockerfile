FROM python:3.6-slim  
COPY ./app.py /initiate/
COPY ./requirements.txt /initiate/
COPY ./iris_trained_model.pkl /initiate/
WORKDIR /initiate/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]

