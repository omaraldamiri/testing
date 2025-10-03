from fastapi import FastAPI, UploadFile, File
from dto import FormData
import numpy as np
import pandas as pd

import pickle

app = FastAPI()

# with open("stacked_meta_model.pkl", "rb") as f:
#     stacked_model = pickle.load(f)
#
# with open("xai_explainer.pkl", "rb") as f:
#     xai_model = pickle.load(f)

label_mapping = ["False Positive", "Planetary Candidate", "Confirmed Exoplanet"]


@app.post("/predict")
async def predict(data: FormData):
    features = [
        data.koi_period,
        data.koi_time0bk,
        data.koi_eccen,
        data.koi_impact,
        data.koi_duration,
        data.koi_depth,
        data.koi_ror,
        data.koi_srho,
        data.koi_prad,
        data.koi_sma,
        data.koi_incl,
        data.koi_teq,
        data.koi_insol,
        data.koi_dor,
    ]

    input_features = np.array([features], dtype=float)

    features_list = input_features.tolist()

    #
    # prediction_num = stacked_model.predict(input_features)[0]
    # prediction_label = label_mapping[int(prediction_num)]
    #
    #
    # explanation_values = xai_model.explain(input_features)

    # return {
    #     "prediction": prediction_label,
    #     "explanation": explanation_values,
    #     "note": "This shows the predicted class and feature-level explanation from the X-AI model."
    # }
    return {
        "recived features ":features_list
    }


@app.post("/predict_csv")
async def predict_csv():
    # Read the CSV into a pandas DataFrame
    # df = pd.read_csv(file.file)
    df = pd.read_csv(r"C:\Users\o1232\OneDrive\Desktop\testing.csv")


    features = df.iloc[0, :14].tolist()  # first row, first 15 columns

    input_features = np.array([features], dtype=float)

    # Convert to list for JSON return
    features_list = input_features.tolist()

    return {"received_features": features_list}
