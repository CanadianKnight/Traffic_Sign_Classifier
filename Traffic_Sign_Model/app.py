from __future__ import division, print_function
# coding=utf-8
import sys
import os
import cv2
from PIL import Image
import glob
import re
import numpy as np

# Keras
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model,model_from_json
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/model_traffic_g.h5'

# Load your trained model
json_file = open('models/model_traffic.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('models/model_traffic.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

thisdict = {
tuple([0]) : 'Speed Limit 20 km/h',
tuple([1]) : 'Speed Limit 30 km/h',
tuple([2]) : 'Speed Limit 50 km/h',
tuple([3]) : 'Speed Limit 60 km/h',
tuple([4]) : 'Speed Limit 70 km/h',
tuple([5]) : 'Speed Limit 80 km/h',
tuple([6]) : 'Speed Limit Ends',
tuple([7]) : 'Speed Limit 100 km/h',
tuple([8]) : 'Speed Limit 120 km/h',
tuple([9]) : 'Overtaking not allowed',
tuple([10]) : 'Overtaking prohibited for trucks',
tuple([11]) : 'Crossroad ahead, side roads to right and left',
tuple([12]) : 'Priority Road Ahead',
tuple([13]) : 'Give way to all Traffic',
tuple([14]) : 'Stop and give way to all Traffic',
tuple([15]) : 'Entry not allowed / forbidden',
tuple([16]) : 'Lorries - Trucks forbidden',
tuple([17]) : 'No entry (one-way traffic)',
tuple([18]) : 'Cars not allowed - prohibited',
tuple([19]) : 'Road ahead curves to the left side',
tuple([20]) : 'Road ahead curves to the right side',
tuple([21]) : 'Double curve ahead, to the left then to the right',
tuple([22]) : 'Poor road surface ahead',
tuple([23]) : 'Slippery road surface ahead',
tuple([24]) : 'Road gets narrow on the right side',
tuple([25]) : 'Roadworks ahead warning',
tuple([26]) : 'Traffic light ahead',
tuple([27]) : 'Warning for pedestrians',
tuple([28]) : 'Warning for children and minors',
tuple([29]) : 'Warning for bikes and cyclists',
tuple([30]) : 'Beware of an icy road ahead',
tuple([31]) : 'Deer crossing in area - road',
tuple([32]) : 'End of entry restriction',
tuple([33]) : 'Turning right compulsory',
tuple([34]) : 'Turning left compulsory',
tuple([35]) : 'Ahead Only',
tuple([36]) : 'Driving straight ahead or turning right mandatory',
tuple([37]) : 'Driving straight ahead or turning left mandatory',
tuple([38]) : 'Pass on right only',
tuple([39]) : 'Pass on left only',
tuple([40]) : 'Direction of traffic on roundabout',
tuple([41]) : 'End of the overtaking prohibition',
tuple([42]) : 'End of the overtaking prohibition for trucks'
}

def model_predict(img_path, model):    

    data=[]
    test_image = cv2.imread(img_path)
    test_image_from_array = Image.fromarray(test_image, 'RGB')
    test_size_image = test_image_from_array.resize((30, 30))
    data.append(np.array(test_size_image))
    x_test = np.array(data)
    x_test = x_test.astype('float32')/255.0
    class_code = model.predict_classes(x_test)
    result = thisdict.get(tuple(class_code))
    return result

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
