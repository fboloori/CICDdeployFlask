from flask import Flask, request, jsonify ,render_template , redirect
import pickle
import numpy as np
from model import MLmodel

app = Flask(__name__)
mlmodel = MLmodel()


@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if request.is_json:
            # Handle JSON input
            print("handle json")
            data = request.get_json()
            features = np.array(data["features"]).reshape(1, -1)
            prediction = mlmodel.predict(features)[0]
            return jsonify({"prediction": prediction})

        else:
            # Handle form data input 
            age = int(request.form['age'])
            sex = request.form.get('sex')
            cp = int(request.form.get('cp'))  # Assuming it's an integer
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])  # Assuming it's an integer
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])  # Assuming it's an integer
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])  # Assuming it's an integer
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])  # Assuming it's an integer

            data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]] , dtype=float)
            my_prediction = mlmodel.predict(data)[0]
            print(data , my_prediction)
            return render_template('result.html', prediction=my_prediction)
      
if __name__=="__main__":
    app.run(debug = True, host="0.0.0.0", port=5000 )      