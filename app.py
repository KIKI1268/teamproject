from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI()

model = joblib.load('clean_model.pkl')

class InputData(BaseModel):
    features: list

@app.get("/")
def read_root():
    with open('EAI6020_FINAL.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.post("/predict")
def predict(data: InputData):
    result = model.predict([data.features])
    return {"prediction": result.tolist()}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000)
