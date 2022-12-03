import random as rd
from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

def BoldReply(msg):
    message = str(msg).lower().strip()



@app.route("/")
def home():
    return render_template("home.html")

    


@app.route("/digit_predict",methods=["POST"])
def predict_image():
    file = request.files['file']
    filename = file.filename
    print("image name : ",filename)


if __name__ == "__main__":
    app.run(debug=True)



