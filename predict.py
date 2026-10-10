"""
Heart Disease Prediction - Inference Script
Loads the trained Scikit-Learn pipeline and predicts heart disease risk for a patient.
"""

import sys
import joblib
import pandas as pd


def load_model(model_path="heart_disease_model.joblib"):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model artifact '{model_path}' not found. Please train or save the model first.")
        sys.exit(1)


def predict_patient(patient_data, model=None):
    if model is None:
        model = load_model()

    df = pd.DataFrame([patient_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "diagnosis": "Heart Disease Detected (High Risk)" if prediction == 1 else "No Heart Disease Detected (Low Risk)",
        "probability": float(probability),
        "confidence_pct": f"{probability * 100:.1f}%" if prediction == 1 else f"{(1 - probability) * 100:.1f}%"
    }


if __name__ == "__main__":
    model = load_model()

    print("=" * 60)
    print("Heart Disease Clinical Prediction System")
    print("=" * 60)

    # Example patient profile 1 (Elevated Risk)
    sample_patient_high_risk = {
        "age": 63,
        "sex": 1,
        "cp": 3,          # Asymptomatic chest pain
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }

    # Example patient profile 2 (Low Risk)
    sample_patient_low_risk = {
        "age": 57,
        "sex": 1,
        "cp": 0,          # Typical angina
        "trestbps": 140,
        "chol": 192,
        "fbs": 0,
        "restecg": 1,
        "thalach": 148,
        "exang": 1,       # Exercise-induced angina
        "oldpeak": 0.4,
        "slope": 1,
        "ca": 0,
        "thal": 1
    }

    print("\n--- Testing Sample Patient 1 (Elevated Risk Profile) ---")
    res1 = predict_patient(sample_patient_high_risk, model)
    print(f"Diagnosis:   {res1['diagnosis']}")
    print(f"Probability: {res1['probability']*100:.1f}% disease risk")

    print("\n--- Testing Sample Patient 2 (Lower Risk Profile) ---")
    res2 = predict_patient(sample_patient_low_risk, model)
    print(f"Diagnosis:   {res2['diagnosis']}")
    print(f"Probability: {res2['probability']*100:.1f}% disease risk")
    print("=" * 60)
