

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.nn import softmax
import io

app = FastAPI()

# Load trained model
model = tf.keras.models.load_model("model.h5")
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



