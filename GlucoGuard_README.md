# GlucoGuard — Intelligent Diabetes Prediction Platform with Risk Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![SMOTE](https://img.shields.io/badge/Imbalanced--learn-SMOTE-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A diabetes risk prediction system built on **Random Forest** with SMOTE-based class balancing, median imputation for biologically invalid zero values, and a **rule-based risk stratification layer** that categorizes predictions into Low / Moderate / High risk tiers. Deployed as a Flask web application with form-based patient input.

---

## 🎯 Key Features

- **Random Forest classifier** (n_estimators=150) trained on Pima Indians Diabetes dataset
- **Median imputation** for biologically invalid zero values in Glucose, BloodPressure, SkinThickness, Insulin, BMI
- **SMOTE oversampling** to handle class imbalance in training data
- **MinMaxScaler** feature normalization
- **Rule-based risk scoring layer** — independently scores 5 clinical features and assigns Low / Moderate / High risk
- **Flask web interface** with form-based patient data entry
- **Modular codebase** — separate train, predict, preprocess, and app modules

---

## 🏗️ System Architecture

```
User Inputs Patient Data (8 clinical features)
                  │
                  ▼
┌─────────────────────────────────────┐
│       MinMaxScaler Transform        │  ← Loaded from scaler.pkl
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│     Random Forest Classifier        │  ← Loaded from diabetes_model.pkl
│       (n_estimators=150)            │    Trained with SMOTE + MinMaxScaler
└─────────────────────────────────────┘
                  │
                  ├─── Binary Prediction: Diabetic / Not Diabetic
                  │
                  ▼
┌─────────────────────────────────────┐
│    Rule-Based Risk Scoring Layer    │  ← Scores Glucose, BMI, Insulin,
│    (Independent of ML model)        │    Age, DiabetesPedigreeFunction
└─────────────────────────────────────┘
                  │
                  ▼
       Risk Level: Low / Moderate / High
                  │
                  ▼
          Flask Web Dashboard
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Model | Random Forest (n_estimators=150) |
| Dataset | Pima Indians Diabetes Dataset |
| Accuracy | ~85% |
| Macro F1-Score | ~0.86 |
| Class Balancing | SMOTE (minority oversampling) |
| Feature Scaling | MinMaxScaler |
| Train/Test Split | 80/20 |

---

## 🧠 Technical Details

### Data Preprocessing
The Pima dataset contains biologically invalid zero values in continuous clinical features. These are replaced with the **column median** (excluding zeros) before training:
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

### Rule-Based Risk Scoring
Beyond the ML prediction, a deterministic rule engine independently scores 5 clinical features:

| Feature | Threshold | Score |
|---------|-----------|-------|
| Glucose | ≥ 140 | +2 |
| Glucose | ≥ 120 | +1 |
| BMI | ≥ 30 | +2 |
| BMI | ≥ 25 | +1 |
| Insulin | ≥ 150 | +2 |
| Insulin | ≥ 100 | +1 |
| Age | ≥ 50 | +2 |
| Age | ≥ 35 | +1 |
| DiabetesPedigreeFunction | ≥ 1.0 | +2 |
| DiabetesPedigreeFunction | ≥ 0.5 | +1 |

**Total Score → Risk Level:**
- 0–2: ✅ Low Risk
- 3–5: ⚠️ Moderate Risk
- 6+: 🛑 High Risk

---

## 📁 Project Structure

```
GlucoGuard/
│
├── app/
│   ├── app.py                      # Flask app + risk scoring logic
│   └── templates/
│       └── index.html              # Patient input form + results display
│
├── src/
│   ├── train_model.py              # Model training pipeline
│   ├── preprocessing.py            # Feature cleaning + scaling utilities
│   ├── predict.py                  # Standalone prediction script
│   └── __init__.py
│
├── data/
│   └── raw/
│       └── diabetes.csv            # Pima Indians Diabetes Dataset
│
├── notebooks/
│   └── diabetes_eda_model.ipynb    # EDA + model training notebook
│
├── models/                         # Saved .pkl files (generated after training)
│   ├── diabetes_model.pkl
│   └── scaler.pkl
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AmudhanManimaran/GlucoGuard.git
cd GlucoGuard
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
cd src
python train_model.py
```
This generates `models/diabetes_model.pkl` and `models/scaler.pkl`.

### 5. Run the Application
```bash
cd app
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## 🚀 Usage

1. Open `http://localhost:5000`
2. Enter the 8 patient clinical features:
   - Pregnancies, Glucose, Blood Pressure, Skin Thickness
   - Insulin, BMI, Diabetes Pedigree Function, Age
3. Click **Predict**
4. View:
   - **Binary prediction** — Diabetic / Not Diabetic
   - **Risk level** — Low / Moderate / High (rule-based)

---

## 📦 Requirements

```
flask>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
joblib>=1.1.0
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Amudhan Manimaran**
- 🌐 Portfolio: [amudhanmanimaran.github.io/Portfolio](https://amudhanmanimaran.github.io/Portfolio/)
- 💼 LinkedIn: [linkedin.com/in/amudhan-manimaran-3621bb32a](https://www.linkedin.com/in/amudhan-manimaran-3621bb32a)
- 🐙 GitHub: [github.com/AmudhanManimaran](https://github.com/AmudhanManimaran)
