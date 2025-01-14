import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
from PIL import Image

# Paths to models
model1_path = r"C:\Users\Dhanvantri\Desktop\Project\models\new_fine_tuned_model.keras"
model2_path = r"C:\Users\Dhanvantri\Desktop\Project\models\fine_tuned_model_high_res.keras"

# Load models
model1 = tf.keras.models.load_model(model1_path)
model2 = tf.keras.models.load_model(model2_path)

# Class names
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]

# Helper function to find a suitable convolutional layer
# Helper function to find a suitable convolutional layer
# Helper function to find the last convolutional layer in the model
def find_last_conv_layer(model):
    # Check if the model has nested layers (e.g., a backbone like MobileNetV2)
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.Sequential, tf.keras.Model)):
            # If it's a nested model, search inside it
            return find_last_conv_layer(layer)
        elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
        elif hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
            print(f"Using fallback layer: {layer.name} ({layer.__class__.__name__})")
            return layer.name

    raise ValueError("No suitable convolutional layer found in the model. Please check the model architecture.")

# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img, img_batch

# Grad-CAM function
# Grad-CAM function
def get_grad_cam_heatmap(model, img_array, class_index):
    # Ensure the model has been called at least once
    if not hasattr(model, 'inputs'):
        # Call the model with a sample input to define inputs/outputs
        dummy_input = tf.zeros((1,) + img_array.shape[1:])
        model(dummy_input)

    # Find the last convolutional layer
    last_conv_layer_name = find_last_conv_layer(model)

    # Create a Grad-CAM model that outputs the feature maps and predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        loss = predictions[:, class_index]

    # Compute the gradients
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Apply gradients to the feature maps
    conv_outputs *= pooled_grads[..., tf.newaxis]

    # Generate the heatmap
    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]

    return heatmap


# Function to overlay heatmap on original image
def overlay_heatmap(heatmap, img, alpha=0.6, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlay_img = cv2.addWeighted(np.array(img), 1 - alpha, heatmap, alpha, 0)
    return overlay_img

# Load a misclassified image
img_path = r"C:\Users\Dhanvantri\Desktop\test\wrong_image\late.jpg"
original_img, img_batch = load_and_preprocess_image(img_path)

# Get predictions from both models
pred1 = model1.predict(img_batch)
pred2 = model2.predict(img_batch)

# Get Grad-CAM heatmaps
heatmap1 = get_grad_cam_heatmap(model1, img_batch, np.argmax(pred1[0]))
heatmap2 = get_grad_cam_heatmap(model2, img_batch, np.argmax(pred2[0]))

# Overlay heatmaps on original image
overlay_img1 = overlay_heatmap(heatmap1, original_img)
overlay_img2 = overlay_heatmap(heatmap2, original_img)

# Save the results
cv2.imwrite(r"C:\Users\Dhanvantri\Desktop\heatmap_model1.jpg", overlay_img1)
cv2.imwrite(r"C:\Users\Dhanvantri\Desktop\heatmap_model2.jpg", overlay_img2)

# Display the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(overlay_img1)
plt.title("Model 1 Heatmap")

plt.subplot(1, 2, 2)
plt.imshow(overlay_img2)
plt.title("Model 2 Heatmap")

plt.show()
