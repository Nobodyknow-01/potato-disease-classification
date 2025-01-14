from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance
import tensorflow as tf
import random

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/my_model.keras")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Preprocessing function without normalization
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")

    # Resize image to match model input size
    image = image.resize((256, 256))

    # Apply random brightness and contrast adjustments
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.9, 1.1))

    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.9, 1.1))

    # Convert to a numpy array without normalization
    image = np.array(image, dtype=np.float32)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    # Adjust the batch size to 32 by padding with zeros if necessary
    img_batch = np.zeros((32, 256, 256, 3), dtype=np.float32)
    img_batch[0] = image  # Place the actual image in the first position

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Custom correction logic for Late Blight misclassification
    if predicted_class == "Early Blight" and confidence > 0.7:
        # Check if the confidence is not extremely high, adjust to Late Blight
        if confidence < 0.9:
            predicted_class = "Late Blight"

    # Class-specific confidence thresholds
    class_thresholds = {
        "Early Blight": 0.65,
        "Late Blight": 0.70,
        "Healthy": 0.80
    }

    threshold = class_thresholds.get(predicted_class, 0.65)

    if confidence < threshold:
        return {
            'class': "Uncertain",
            'confidence': float(confidence)
        }

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=80)
