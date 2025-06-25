from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import torch
import pathlib

class PosixPathFix(pathlib.PosixPath):
    pass
pathlib.WindowsPath = PosixPathFix

app = FastAPI()

model = None
is_torch_model = False

try:
    model = joblib.load("garbage_classifier.pkl")
    print("Model loaded successfully (joblib/scikit-learn format).")
except Exception as e:
    print(f"joblib load failed: {e}")
    try:
        model = torch.load("garbage_classifier.pkl", map_location=torch.device("cpu"))
        model.eval()
        is_torch_model = True
        print("Model loaded successfully (PyTorch format).")
    except Exception as e2:
        raise RuntimeError(f"Failed to load model with both joblib and torch: {e2}")

class GarbageRequest(BaseModel):
    features: list  

@app.get("/")
def read_root():
    return {"message": "Garbage Classification API is running"}

@app.post("/predict/")
def predict(request: GarbageRequest):
    try:
        features = np.array(request.features).reshape(1, -1)

        if is_torch_model:
            tensor_input = torch.tensor(features, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(tensor_input)
                prediction = outputs.numpy().tolist() if hasattr(outputs, "numpy") else outputs.tolist()
        else:
            prediction = model.predict(features).tolist()

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

