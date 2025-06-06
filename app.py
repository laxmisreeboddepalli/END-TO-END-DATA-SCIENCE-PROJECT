from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    features = np.array([values])
    prediction = model.predict(features)[0]
    return render_template("index.html", prediction_text=f"Predicted Class: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
