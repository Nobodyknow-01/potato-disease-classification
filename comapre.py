import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Paths to both models
model1_path = r"C:\Users\Dhanvantri\Desktop\Project\models\fine_tuned_model.keras"
model2_path = r"C:\Users\Dhanvantri\Desktop\Project\models\fine_tuned_model_high_res.keras"

# Load both models
model1 = tf.keras.models.load_model(model1_path)
model2 = tf.keras.models.load_model(model2_path)

# Class names
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]

# Path to test images (replace with your actual path)
TEST_IMAGES_DIR = r"C:\Users\Dhanvantri\Desktop\test_images"

# Function to load and preprocess images
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_batch

# Initialize counters
correct_model1 = 0
correct_model2 = 0

total_images = len(os.listdir(TEST_IMAGES_DIR))

# Iterate over images in the test directory
for img_file in os.listdir(TEST_IMAGES_DIR):
    img_path = os.path.join(TEST_IMAGES_DIR, img_file)
    img_batch = load_and_preprocess_image(img_path)

    # Get predictions from both models
    pred1 = model1.predict(img_batch)
    pred2 = model2.predict(img_batch)

    # Get the predicted class and confidence for both models
    class1 = CLASS_NAMES[np.argmax(pred1[0])]
    confidence1 = np.max(pred1[0])

    class2 = CLASS_NAMES[np.argmax(pred2[0])]
    confidence2 = np.max(pred2[0])

    # Check if the prediction is correct (all images are Early Blight)
    if class1 == "Early Blight":
        correct_model1 += 1
    if class2 == "Early Blight":
        correct_model2 += 1

    # Display the results
    print(f"Image: {img_file}")
    print(f"Model 1 Prediction: {class1} (Confidence: {confidence1:.2f})")
    print(f"Model 2 Prediction: {class2} (Confidence: {confidence2:.2f})")
    print("-" * 50)

# Calculate and display scores
print("\nFinal Scores:")
print(f"Model 1 Accuracy: {correct_model1}/{total_images} ({(correct_model1 / total_images) * 100:.2f}%)")
print(f"Model 2 Accuracy: {correct_model2}/{total_images} ({(correct_model2 / total_images) * 100:.2f}%)")
