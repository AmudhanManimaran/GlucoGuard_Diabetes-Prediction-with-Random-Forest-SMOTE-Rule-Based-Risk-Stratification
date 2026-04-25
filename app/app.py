from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('../models/diabetes_model.pkl')
scaler = joblib.load('../models/scaler.pkl')


# ------------------------
# Risk Level Function 🔍
# ------------------------
def assess_risk_level(input_data):
    pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age = input_data

    risk_score = 0

    if glucose >= 140:
        risk_score += 2
    elif glucose >= 120:
        risk_score += 1

    if bmi >= 30:
        risk_score += 2
    elif bmi >= 25:
        risk_score += 1

    if insulin >= 150:
        risk_score += 2
    elif insulin >= 100:
        risk_score += 1

    if age >= 50:
        risk_score += 2
    elif age >= 35:
        risk_score += 1

    if dpf >= 1:
        risk_score += 2
    elif dpf >= 0.5:
        risk_score += 1

    if risk_score <= 2:
        return "✅ Low Risk"
    elif risk_score <= 5:
        return "⚠️ Moderate Risk"
    else:
        return "🛑 High Risk"


# ------------------------
# Route & Logic
# ------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    risk_level = None
    if request.method == 'POST':
        try:
            input_data = [
                float(request.form['pregnancies']),
                float(request.form['glucose']),
                float(request.form['blood_pressure']),
                float(request.form['skin_thickness']),
                float(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['dpf']),
                float(request.form['age'])
            ]
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)

            result = "🩸 Positive for Diabetes" if prediction[0] == 1 else "✅ Not Diabetic"
            risk_level = assess_risk_level(input_data)

        except:
            result = "❌ Invalid input. Please enter valid numbers."
            risk_level = None

    return render_template('index.html', result=result, risk_level=risk_level)


if __name__ == '__main__':
    app.run(debug=True)
