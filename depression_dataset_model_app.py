from flask import Flask, request, render_template, flash, session
import os
import joblib
import numpy as np
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.secret_key = os.urandom(12)

# Load trained model and prepare imputer
model_path = "/home/nasir-hussain/Pictures/Depression_Detection_Using_Machine_Learning/best_model_RandomForest.pkl"
model = joblib.load(model_path)
imputer = SimpleImputer(strategy='mean')


@app.route('/')
def root():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('index.html')


@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'admin' and request.form['username'] == 'admin':
        session['logged_in'] = True
    else:
        flash('wrong password!')
    return root()


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return root()


@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get form inputs
        q1 = int(request.form['a1'])
        q2 = int(request.form['a2'])
        q3 = int(request.form['a3'])
        q4 = int(request.form['a4'])
        q5 = int(request.form['a5'])
        q6 = int(request.form['a6'])
        q7 = int(request.form['a7'])
        q8 = int(request.form['a8'])
        q9 = int(request.form['a9'])
        q10 = int(request.form['a10'])

        input_values = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]

        # Add default values for extra features used during training
        hour = 12
        dayofweek = 2
        period_name = 1

        features = input_values + [hour, dayofweek, period_name]
        features = np.array(features).reshape(1, -1)
        features = imputer.fit_transform(features)

        prediction = model.predict(features)

        # Map prediction to result text
        result_map = {
            0: 'Your Depression test result : No Depression',
            1: 'Your Depression test result : Mild Depression',
            2: 'Your Depression test result : Moderate Depression',
            3: 'Your Depression test result : Moderately severe Depression',
            4: 'Your Depression test result : Severe Depression'
        }

        result = result_map.get(prediction[0], 'Could not determine result.')
        print(result)
        return render_template("result.html", result=result)


    except Exception as e:
        return f"❌ Error in prediction: {str(e)}", 500


if __name__ == '__main__':
    app.run(port=5987, host='0.0.0.0', debug=True)
