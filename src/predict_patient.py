import pandas as pd

def predict_patient_data(model, scaler, patient_data):
    df = pd.DataFrame([patient_data])
    scaled = scaler.transform(df)
    pred = model.predict(scaled)
    proba = model.predict_proba(scaled)
    return pred[0], proba[0]
