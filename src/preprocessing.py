import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_scale_data(df):
    features_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace 0 with column mean (excluding zeros)
    for feature in features_to_clean:
        non_zero_values = df[df[feature] != 0][feature]
        mean_value = non_zero_values.mean()
        df[feature] = df[feature].replace(0, mean_value)

    # Split features and labels
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
