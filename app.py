#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
from flask import Flask, request

model = None

def load_model():
    global model
    # model variable refers to the global variable 
    with open('iris_trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

# all flask functions require an application instance, the webserver passes
# requests from clients to the application object for handlingg
# The applicaition instance is an object of class Flask usually passing
# __name__ to the flask constructor is the only required arguement

# Now how do we handle the request? We create routes and view functions. The
# app.route decorator allows us to define the route '/' which is exposed by the
# applicaiton instance, so when a client sends a request to the '/' (or url)
# it will call the view function below (home_endpoint) and return the function
# output

app = Flask(__name__)

@app.route('/')
def home_endpoint():
    return "Hello World!"


@app.route('/predict', methods=['POST'])
def get_prediction():
    # works only for a single sample
    if request.method == 'POST': # The endpoint accepts a ‘POST’ request
        data = request.get_json() # get data posted as json
        data = np.array(data)[np.newaxis, :] # converts shape from (4, ) to (1, 4)
        prediction = model.predict(data) # runs globally loaded model on  data
    return str(prediction[0])

if __name__ == '__main__':
    load_model() # load model at the beginning, once only
    app.run(host='0.0.0.0', port=5000)


