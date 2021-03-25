#Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
from PIL import Image
import os, sys
import cv2
import numpy as numpy
import matplotlib.pyplot as plt


# Create flask instance
app = Flask(__name__)

class_labels = ['barretts', 'bbps-0-1', 'bbps-2-3', 'cecum', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a', 'esophagitis-b-d', 'impacted-stool', 'polyps', 'pylorus', 'retroflex-rectum', 'retroflex-stomach', 'ulcerative-colitis-grade-0-1', 'z-line'] 

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'avi']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.Graph()

# Function to load and prepare the image in right shape
def read_image(filename):
    # Load the image
    img = load_img(filename, target_size=(224, 224, 3))
    # Convert the image to array
    img = img_to_array(img)
    # Reshape the image into a sample of 1 channel
    img = np.expand_dims(img, axis=0)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('static/images', filename)
                file.save(file_path)
                sec = 0
                frameRate = 5 #//it will capture image in each 0.5 second
                count=1
                vidcap = cv2.VideoCapture(file_path)
                def getFrame(sec):
                    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
                    hasFrames,image = vidcap.read()
                    if hasFrames:
                        cv2.imwrite(os.path.join('static/images', str(000000) + str(count) + '.jpg'), image)     # save frame as JPG file
                        return hasFrames
                success = getFrame(sec)

                while success:
                    count = count + 1
                    sec = sec + frameRate
                    sec = round(sec, 2)
                    success = getFrame(sec)
                i = 1
                lst = []
                userimage = []
                while i < count:
                    file_path1 = os.path.join('static/images', str(000000) + str(i) + '.jpg')
                    img = read_image(file_path1)
                # Predict the class of an image

                #with graph.as_default():
                    model1 = load_model('kvasir.h5')
                    class_prediction = np.argmax(model1.predict(img))
                    product = (class_labels[class_prediction])
                    lst.append(product)
                    userimage.append(file_path1)
                    i = i + 1
                user_image1 = userimage[1]
                user_image2 = userimage[2]
                user_image3 = userimage[3]
                user_image4 = userimage[4]
                lst = list(set(lst))
                name = request.form['name']
                dob = request.form['dob']
                gp = request.form['gp']
                hn = request.form['hn']
                dop = request.form['dop']
                ep = request.form['ep']
                nurses = request.form['nurses']
                return render_template('predict.html', lst = lst, user_image1 = user_image1, user_image2 = user_image2, user_image3 = user_image3, user_image4 = user_image4, name = name, dob = dob, gp = gp, hn = hn, dop = dop, ep = ep, nurses = nurses)
        
    return render_template('predict.html')

if __name__ == "__main__":
    init()
    app.run()