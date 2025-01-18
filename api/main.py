from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Initialize FastAPI
app = FastAPI()

# CORS settings
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


# Load the trained model
MODEL = tf.keras.models.load_model("../models/new_fine_tuned_model.keras")

CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]



@app.get("/")
def read_root():
    return {"message": "Welcome to the Potato Disease Classification API"}

@app.get("/ping")
async def ping():
    return "Hello, I am alive"



# Read and preprocess the image
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))  # Resize to the model's expected input size
    image = np.array(image) / 255.0  # Normalize pixel values
    return image





# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Add batch dimension

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)

    }


# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8001)


