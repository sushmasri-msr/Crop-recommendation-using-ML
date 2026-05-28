from flask import Flask, render_template, request, redirect
import numpy as np
import os
import pickle

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "randomforest.pkl"), "rb"))
label_encoder = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/login_success", methods=["POST"])
def login_success():
    return redirect("/crop")


@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/signup_success", methods=["POST"])
def signup_success():
    return redirect("/crop")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/crop", methods=["GET", "POST"])
def crop():
    prediction = None

    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        pred_encoded = model.predict(input_data)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]

    return render_template("crop.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)