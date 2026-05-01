import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.nn import softmax
import io
import uvicorn

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.h5")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully")
    yield

app = FastAPI(lifespan=lifespan)

class_names = ['Open Eye', 'Sleepy Eye']
IMG_SIZE = (150, 150)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probs = softmax(predictions).numpy()[0]

    results = [
        {"class": class_names[i], "probability": float(probs[i])}
        for i in range(len(class_names))
    ]

    return JSONResponse(content={"predictions": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port)