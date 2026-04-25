import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# 🔹 Load dataset
df = pd.read_csv('../data/raw/diabetes.csv')

# 🔹 Replace 0s with median in key columns
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())

# 🔹 Feature/target split
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 🔹 MinMax Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Train/test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔹 SMOTE (Oversample minority class)
smote = SMOTE(random_state=42)
Xtrain_sm, ytrain_sm = smote.fit_resample(Xtrain, ytrain)

# 🔹 Train Random Forest
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(Xtrain_sm, ytrain_sm)

# 🔹 Evaluate
y_pred = model.predict(Xtest)
acc = accuracy_score(ytest, y_pred)

print("🎯 Accuracy:", acc)
print("📊 Confusion Matrix:\n", confusion_matrix(ytest, y_pred))
print("📋 Classification Report:\n", classification_report(ytest, y_pred))

# 🔹 Save model and scaler
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/diabetes_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

print("✅ Model and scaler saved successfully.")
