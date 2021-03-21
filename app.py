from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'models/malaria_detector.h5'

model = load_model(MODEL_PATH)
#model._make_predict_function()  # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(130, 130, 3))

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    pred = np.argmax(preds, axis=1)
    return pred


@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']


        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        pred = model_predict(file_path, model)
        os.remove(file_path)

        str1 = 'Malaria Parasitized'
        str2 = 'Normal'
        if pred[0] == 0:
            return str1
        else:
            return str2
    return None




if __name__ == '__main__':
    app.run()
