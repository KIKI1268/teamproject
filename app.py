
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastai.learner import load_learner
from fastai.vision.all import PILImage
from io import BytesIO
from PIL import Image
import pathlib  

class PosixPathFix(pathlib.PosixPath):
    pass
pathlib.WindowsPath = PosixPathFix

app = FastAPI()

model = load_learner('garbage_classifier.pkl')

@app.get("/")
async def root():
    return {"message": "Garbage Classification API is running. Use /predict to upload images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        img = PILImage.create(image)

        pred, _, probs = model.predict(img)

        return {
            "prediction": str(pred),
            "probabilities": probs.tolist()
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
