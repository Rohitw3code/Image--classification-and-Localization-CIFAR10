from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras

model = keras.models.load_model('imgclassification.h5')

app = Flask(__name__)

labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()

def BoldReply(msg):
    message = str(msg).lower().strip()



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
        imgdata = plt.imread(f.filename)
        n = np.array(imgdata)/255
        print(n.shape)
        p = n.reshape(1, 32, 32, 3)
        predicted_label = labels[model.predict(p).argmax()]
        print("predicted label is {}".format(predicted_label))
        return "predicted label is {}".format(predicted_label)

if __name__ == "__main__":
    app.run(debug=True)



