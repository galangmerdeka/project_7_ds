import os
import pandas as pd
import joblib
import data_models as dm

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "classifier", "model_rf.joblib")
CLASSIFIER = joblib.load(model_path)

def predict(data: dm.Applicant) -> str:
    inp = format_data(data)
    raw_pred = CLASSIFIER.predict(inp)
    prediction = "Yes" if (raw_pred == 1) else "No"
    return prediction

def format_data(data: dm.Applicant) -> pd.DataFrame:
    dict_data = data.dict()
    inp = pd.DataFrame([dict_data], columns=dict_data.keys())
    return inp