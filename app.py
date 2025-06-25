from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

try:
    model = joblib.load("garbage_classifier.pkl")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise e

class GarbageRequest(BaseModel):
    features: list  # Example: [0.1, 0.5, 3.2]

@app.get("/")
def read_root():
    return {"message": "Garbage Classification API is up and running"}

@app.post("/predict/")
def predict(request: GarbageRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


