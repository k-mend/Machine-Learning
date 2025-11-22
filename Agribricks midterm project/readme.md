# Crop and Rain Prediction API

This project provides a **FastAPI-based API** to predict the probability of rain, estimate precipitation amount, and recommend crops suitable for given weather conditions. The project was originally developed in Jupyter notebooks, and the final trained models are packaged for deployment.

## Problem Statement

This project uses **historical weather and crop data to build predictive models** that forecast rainfall (both probability and amount) and recommend suitable crops based on agro-climatic conditions. It addresses the challenge of agricultural planning under uncertain weather by combining **XGBoost and RandomForest models** for rain prediction with a crop recommendation system based on temperature and rainfall tolerances. The goal is to provide a **reliable, data-driven decision support tool for farmers**. The models and data are based on **Kenya's weather dataset**.

---

## Dataset

### NASA POWER API
Base URL: `https://power.larc.nasa.gov/api/temporal/daily/point`

#### Parameters
```python
# Historical date range for training
START_DATE = "19930101"
END_DATE = "20251031"

# Weather parameters from NASA POWER
# PRECTOTCORR - Precipitation (mm/day)
# T2M - Temperature at 2m (°C)
# RH2M - Relative Humidity at 2m (%)
# ALLSKY_SFC_SW_DWN - All Sky Surface Shortwave Downward Irradiance (W/m²) - proxy for sunshine
PARAMETERS = ["PRECTOTCORR", "T2M", "RH2M", "ALLSKY_SFC_SW_DWN"]
```

#### Locations
```python
LOCATIONS = {
    "highlands_humid_nyeri": {"name": "Nyeri (Highlands, Humid)", "latitude": -0.4167, "longitude": 36.9500},
    "upper_midlands_kitale": {"name": "Kitale (Upper Midlands, High Potential)", "latitude": 1.0167, "longitude": 35.0000},
    "lower_midlands_semiarid_machakos": {"name": "Machakos (Lower Midlands, Semi-Arid)", "latitude": -1.5167, "longitude": 37.2667},
    "coastal_lowlands_malindi": {"name": "Malindi (Coastal Lowlands, Humid)", "latitude": -3.2236, "longitude": 40.1300},
    "arid_lowlands_lodwar": {"name": "Lodwar (Arid Lowlands, Arid)", "latitude": 3.1191, "longitude": 35.5973}
}
```

### Ecocrop Dataset

You can download the Ecocrop dataset from their GitHub repo: [EcoCrop_DB.csv](https://github.com/OpenCLIM/ecocrop/blob/main/EcoCrop_DB.csv)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Setup and Installation](#setup-and-installation)
5. [Running with Docker](#running-with-docker)
6. [API Endpoints](#api-endpoints)
7. [Models](#models)
8. [Usage Example](#usage-example)
9. [License](#license)

---

## Project Overview

The project consists of:

* **Weather and crop datasets**: Used to train models and recommend crops.
* **Machine learning models**:

  * `rain_classifier.pkl`: Classifies if it will rain (binary classification).
  * `rain_regressor.pkl`: Predicts the amount of precipitation (regression).
  * `dict_vectorizer.pkl`: Preprocesses weather features for the models.
  * `ecocrop_df.pkl`: Crop database with temperature and rainfall ranges.
* **FastAPI API**: Provides endpoints for predictions and crop recommendations.
* **Dockerfile**: Containerizes the API for deployment.

**Why FastAPI over Flask:**

FastAPI is chosen because it offers **high performance comparable to Node.js**, automatic **data validation** via Pydantic models, and **built-in interactive API documentation** (Swagger UI and ReDoc). Unlike Flask, FastAPI is **async-friendly**, which allows handling multiple requests concurrently, improving performance for APIs that serve machine learning predictions. FastAPI also provides **type hints and auto-generated docs**, reducing boilerplate and potential errors.

**How it works:**

1. Input JSON data is sent to the API endpoints (`/predict` or `/recommend_crops`).
2. FastAPI validates and parses the input using **Pydantic models**.
3. The validated input is converted into the format expected by the trained ML models.
4. The ML models predict rain probability, precipitation, or suitable crops.
5. FastAPI returns the predictions in JSON format.

---

## Project Structure
```
├── app
│   ├── main.py
│   └── __pycache__
├── data
│   ├── cleaned_ecocrop.csv
│   └── merged_aez_weather.csv
├── Dockerfile
├── models
│   ├── dict_vectorizer.pkl
│   ├── ecocrop_df.pkl
│   ├── rain_classifier.pkl
│   └── rain_regressor.pkl
├── notebooks
│   ├── anaconda_projects
│   ├── tests
│   └── training.ipynb
├── readme.md
├── requirements.txt
├── Screenshots
│   ├── Api running.png
│   ├── Docker build.png
│   ├── DOCKER RESPONSE.png
│   ├── rainfall_probability_and _amount.png
│   ├── recommended_crops_response.png
│   └── weather_docker response.png
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── lib64 -> lib
    └── pyvenv.cfg
```

---

## Requirements

* Python 3.9+
* pip
* Docker (optional, for containerized deployment)

**Python dependencies**:
```txt
fastapi
uvicorn
scikit-learn
xgboost
pandas
numpy
```

Optional (for plotting in notebooks):
```txt
matplotlib
seaborn
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/k-mend/machine-learning.git
cd 'machine learning'
cd 'Agribricks midterm project'
```

2. Ensure all model files are in the `models/` directory:

* `dict_vectorizer.pkl`
* `rain_classifier.pkl`
* `rain_regressor.pkl`
* `ecocrop_df.pkl`

3. (Optional) Run the FastAPI app locally:
```bash
uvicorn app.main:app --reload
```

![API Running Locally](Screenshots/Api%20running.png)

The API will be available at: `http://127.0.0.1:8000`

---

## Running with Docker

Build the Docker image:
```bash
docker build -t crop-rain-api .
```

![Docker Build](Screenshots/Docker%20build.png)

Run the container:
```bash
docker run -d -p 8000:8000 crop-rain-api
```

![Docker Running](Screenshots/DOCKER%20RESPONSE.png)

The API will be accessible at: `http://localhost:8000`

---

## API Endpoints

| Endpoint           | Method | Description                                                                                                                                                  |
| ------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/predict`         | POST   | Predicts rain probability and precipitation for a given weather sample. Input: JSON weather features. Output: `{"rain_prob": float, "precipitation": float}` |
| `/recommend_crops` | POST   | Returns a list of crops suitable for given site temperature and rainfall. Input: `{"temperature": float, "rainfall": float}`                                 |

**Example JSON for `/predict`**:
```json
{
  "t2m": 25.0,
  "rh2m": 80.0,
  "allsky_sfc_sw_dwn": 300.0,
  "month": 6,
  "aez": "AEZ1"
}
```

**Example JSON for `/recommend_crops`**:
```json
{
  "temperature": 25.0,
  "rainfall": 100.0
}
```

---

## Models

* **Rain Classifier (`rain_classifier.pkl`)**: RandomForestClassifier trained on historical weather data to predict if it will rain.
* **Rain Regressor (`rain_regressor.pkl`)**: RandomForestRegressor predicting precipitation amount.
* **DictVectorizer (`dict_vectorizer.pkl`)**: Converts JSON input into numeric feature vectors for the models.
* **Ecocrop Data (`ecocrop_df.pkl`)**: Crop database for recommending crops based on temperature and rainfall tolerance.

All models are pre-trained and saved in the `models/` folder.

---

## Usage Example

### 1. Predicting Rain Probability and Precipitation

Start the API (locally or via Docker), then send a POST request to `/predict`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"t2m":25,"rh2m":80,"allsky_sfc_sw_dwn":300,"month":6,"aez":"AEZ1"}'
```

**Response:**

![Rain Prediction Response](Screenshots/rainfall_probability_and%20_amount.png)
```json
{
  "rain_prob": 0.78,
  "precipitation": 12.3
}
```

### 2. Recommending Crops

Send a POST request to `/recommend_crops`:
```bash
curl -X POST "http://127.0.0.1:8000/recommend_crops" \
-H "Content-Type: application/json" \
-d '{"temperature":25,"rainfall":100}'
```

**Response:**

![Crop Recommendation Response](Screenshots/recommended_crops_response.png)
```json
["Maize", "Rice", "Soybean"]
```

### 3. Testing with Python Script

You can also test the API using a Python script:
```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "t2m": 25.0,
    "rh2m": 80.0,
    "allsky_sfc_sw_dwn": 300.0,
    "month": 6,
    "aez": "AEZ1"
}

response = requests.post(url, json=data)
print(response.json())
```

![Weather Docker Response](Screenshots/weather_docker%20response.png)

---

## Cloud Deployment
This service has been deployed on Render 
![Render deployment logs](Screenshots/render_runtime.png)

Here is the link to whic the application is deployed but due to large size of the models i trained (over 428mb) render could not host my application so it was brought down. Here is the link for records [render link](https://machine-learning-eqrn.onrender.com/)


## License

This project is licensed under MIT License.
