from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dto import FormData
import numpy as np
import pandas as pd
import pickle

app = FastAPI()

# Enable CORS (so your HTML+JS on localhost:3000 can call backend on 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in prod, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uncomment if you load models
# with open("stacked_meta_model.pkl", "rb") as f:
#     stacked_model = pickle.load(f)
#
# with open("xai_explainer.pkl", "rb") as f:
#     xai_model = pickle.load(f)

label_mapping = ["False Positive", "Planetary Candidate", "Confirmed Exoplanet"]


@app.post("/predict")
async def predict(data: FormData):
    """
    Accepts JSON data from frontend form and returns processed features
    """
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
    print("excuted")




    input_features = np.array([features], dtype=float)

    return {"classification": "test", "confidence_score": 0.95 , "features":features }



@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    try:

        df = pd.read_csv(file.file)  # Read uploaded file
        features = df.iloc[0, :14].tolist()  # SECOND row (index 1), first 14 columns
        input_features = np.array([features], dtype=float)
        # For now, return dummy data (replace with actual prediction later)
        return {
            "classification": "Planetary Candidate",  # Match what frontend expects
            "confidence_score": 0.85,                  # Match what frontend expects
            "received_features": input_features.tolist()  # Optional: for debugging
        }
    except Exception as e:
        return {
            "classification": "Error",
            "confidence_score": 0.0,
            "error": str(e)
        }
