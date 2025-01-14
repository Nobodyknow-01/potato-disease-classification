import os
from PIL import Image
from collections import Counter

def check_class_distribution_and_image_sizes(base_path):
    # Initialize counters and lists
    class_counts = {}
    image_sizes = []  # Ensure this is initialized outside the loop

    print("Checking paths...")

    # Recursively traverse all subdirectories under base_path
    for root, dirs, files in os.walk(base_path):
        print(f"Checking directory: {root}")  # Log the directory being checked
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"Checking file: {file_path}")  # Log each file being checked

            # Only process image files (with extensions .jpg, .jpeg, .png, .bmp, .tiff)
            if filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                try:
                    with Image.open(file_path) as img:
                        img_size = img.size
                        image_sizes.append(img_size)
                        # Update class count based on the directory name
                        class_name = os.path.basename(root)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                except Exception as e:
                    print(f"Error opening image {filename}: {e}")

    print("\nClass Distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")

    # Most common image sizes
    if image_sizes:
        most_common_sizes = Counter(image_sizes).most_common(3)  # top 3 sizes
        print("\nMost common image sizes:")
        for size, count in most_common_sizes:
            print(f"Size: {size}, Count: {count}")
    else:
        print("\nNo valid images found to analyze.")

# Run the function with the base path to your dataset
base_path = r"C:\Users\Dhanvantri\Desktop\New folder (4)\PlantVillage\Potato___Late_blight"
check_class_distribution_and_image_sizes(base_path)
