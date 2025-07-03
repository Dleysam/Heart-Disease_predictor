
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load('heart Disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(data)[0]

    result = "Positive (Heart Disease)" if prediction == 1 else "Negative (No Heart Disease)"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
