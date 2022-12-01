import random as rd
from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

def BoldReply(msg):
    message = str(msg).lower().strip()



@app.route("/")
def home():
    return render_template("home.html")

    

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    reply  = BoldReply(userText)
    return reply


if __name__ == "__main__":
    app.run(debug=True)



