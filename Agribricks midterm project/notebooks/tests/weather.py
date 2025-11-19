import requests
import pandas as pd

weather = pd.read_csv("../../data/merged_aez_weather.csv")

# Pick a sample row
sample_idx = 500  # or min(50, len(weather)-1)
sample_row = weather.iloc[sample_idx]

# Map the dataframe columns to the API's expected keys
sample_dict_api = {
    "T2M": float(sample_row["T2M"]),
    "RH2M": float(sample_row["RH2M"]),
    "ALLSKY_SFC_SW_DWN": float(sample_row["ALLSKY_SFC_SW_DWN"]),
    "month": 6,   # or extract month from a date column if available
    "AEZ": "Highlands (Humid)"
}


# URL of the running FastAPI endpoint
url = "http://127.0.0.1:8003/predict_weather"

# Send POST request
response = requests.post(url, json=sample_dict_api)

# Print results
if response.status_code == 200:
    print("Sample index:", sample_idx)
    print("Input features:", sample_dict_api)
    print("API prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)
