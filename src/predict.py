import joblib
import numpy as np

# Load model and scaler
model = joblib.load('../models/diabetes_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# Sample input (replace with user input later)
# Format: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
sample_input = np.array([[5, 130, 70, 28, 100, 32.0, 0.5, 35]])

# Scale input
scaled_input = scaler.transform(sample_input)

# Predict
prediction = model.predict(scaled_input)

# Output
if prediction[0] == 1:
    print("✅ Prediction: The person is likely to have diabetes.")
else:
    print("❌ Prediction: The person is not likely to have diabetes.")
