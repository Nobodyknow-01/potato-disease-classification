from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

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

# Load both models
model1 = tf.keras.models.load_model(r"C:\Users\Dhanvantri\Desktop\Project\models\fine_tuned_model.keras")
model2 = tf.keras.models.load_model(r"C:\Users\Dhanvantri\Desktop\Project\models\fine_tuned_model_high_res.keras")

# Class names
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.15

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Preprocessing function
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    # Get predictions from both models
    predictions1 = model1.predict(image)
    predictions2 = model2.predict(image)

    # Get predicted class and confidence for both models
    predicted_class1 = CLASS_NAMES[np.argmax(predictions1[0])]
    confidence1 = np.max(predictions1[0])

    predicted_class2 = CLASS_NAMES[np.argmax(predictions2[0])]
    confidence2 = np.max(predictions2[0])

    # Updated prediction logic
    if predicted_class1 == predicted_class2:
        final_class = predicted_class1
        final_confidence = (confidence1 + confidence2) / 2  # Average confidence
    else:
        # Compare confidence scores and check the difference
        confidence_diff = abs(confidence1 - confidence2)
        if confidence_diff >= CONFIDENCE_THRESHOLD:
            if confidence1 > confidence2:
                final_class = predicted_class1
                final_confidence = confidence1
            else:
                final_class = predicted_class2
                final_confidence = confidence2
        else:
            # If confidence scores are close, prioritize the class with "Late Blight" prediction
            final_class = "Late Blight" if "Late Blight" in [predicted_class1, predicted_class2] else "Early Blight"
            final_confidence = max(confidence1, confidence2)

    return {
        'class': final_class,
        'confidence': float(final_confidence),
        'model1_prediction': predicted_class1,
        'model2_prediction': predicted_class2,
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)
