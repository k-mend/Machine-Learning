from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# -------------------------
# Load models and data
# -------------------------
with open("models/dict_vectorizer.pkl", "rb") as f:
    dv = pickle.load(f)
with open("models/rain_classifier.pkl", "rb") as f:
    rain_clf = pickle.load(f)
with open("models/rain_regressor.pkl", "rb") as f:
    rain_reg = pickle.load(f)
with open("models/ecocrop_df.pkl", "rb") as f:
    ecocrop = pickle.load(f)

features = list(dv.feature_names_)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Weather & Crop API")

# -------------------------
# Request models
# -------------------------
class WeatherInput(BaseModel):
    T2M: float
    RH2M: float
    ALLSKY_SFC_SW_DWN: float
    month: int
    AEZ: str

class SiteInput(BaseModel):
    site_temp: float
    site_rain: float
    tolerance: float = 0.0

# -------------------------
# Crop recommendation
# -------------------------
def recommend_crops(site_temp, site_rain, tol=0.0):
    candidates = []
    for _, row in ecocrop.iterrows():
        tmin, tmax = row.get('tmin', np.nan), row.get('tmax', np.nan)
        rmin, rmax = row.get('rmin', np.nan), row.get('rmax', np.nan)
        if np.isnan([tmin, tmax, rmin, rmax]).any(): 
            continue
        if tmin - tol <= site_temp <= tmax + tol and rmin - tol <= site_rain <= rmax + tol:
            candidates.append(row.get('scientificname', row.get('comname', None)))
    return list(filter(None, candidates))

# -------------------------
# Endpoints
# -------------------------
@app.post("/predict_weather")
def predict_weather(data: WeatherInput):
    X = dv.transform([data.dict()])
    prob = float(rain_clf.predict_proba(X)[0][1])
    amt = float(rain_reg.predict(X)[0])
    return {"rain_probability": prob, "predicted_precipitation": amt}

@app.post("/recommend_crops")
def recommend(data: SiteInput):
    crops = recommend_crops(data.site_temp, data.site_rain, data.tolerance)
    return {"recommended_crops": crops}
