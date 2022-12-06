from flask import Flask, render_template, request
from tensorflow import keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import io
import PIL
import cv2
import xmltodict
from base64 import b64encode
from PIL import Image
import random
from PIL import ImageDraw
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import plot_model
from io import BytesIO
import base64

import pickle

model = keras.models.load_model('imgclassification.h5')
rmodel = tf.keras.models.load_model("image_local_model.h5", compile=False)

app = Flask(__name__)


with open('file.pkl', 'rb') as file:
    encoder = pickle.load(file)

labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()


def predictDrawbox(model, image, le):
    img = tf.cast(np.expand_dims(image, axis=0), tf.float32)
    # prediction
    predict = model.predict(img)
    pred_box = predict[..., 0:4] * 228
    x = pred_box[0][0]
    y = pred_box[0][1]
    w = pred_box[0][2]
    h = pred_box[0][3]
    # get class name
    trans = le.inverse_transform(predict[..., 4:])

    img = BytesIO()
    # im = PIL.Image.fromarray(image)
    # draw = ImageDraw.Draw(im)
    # draw.rectangle([x, y, w, h], outline='red')
    # plt.xlabel(trans[0])

    file_object = io.BytesIO()
    img= Image.fromarray(image.astype('uint8'))
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, w, h], outline='red')
    img.save(file_object, 'PNG')
    base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')

    return base64img,trans[0]


@app.route("/")
def home():
    return render_template("home.html")

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        imgdata = plt.imread(f.filename)
        n = np.array(imgdata)/255
        print(n.shape)
        p = n.reshape(1, 32, 32, 3)
        predicted_label = labels[model.predict(p).argmax()]
        print("predicted label is {}".format(predicted_label))
        return "predicted label is {}".format(predicted_label)

@app.route('/imageLocalUploader', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        imgdata = cv2.resize(cv2.imread(f.filename), (228, 228))
        rimg = np.array(imgdata)
        finalimg = predictDrawbox(rmodel,rimg,encoder)
        return render_template("plot.html",displayimage=finalimg[0],label=finalimg[1])


if __name__ == "__main__":
    app.run(debug=True)



