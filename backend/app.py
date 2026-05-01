import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.nn import softmax
import io
import uvicorn

app = FastAPI()

# Load trained model
model = None

@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model("backend/model.h5", compile=False)
    print("Model loaded successfully")

class_names = ['Open Eye', 'Sleepy Eye']
IMG_SIZE = (150, 150)

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