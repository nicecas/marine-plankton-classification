import os
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Set the source data folder and the destination folder path
source_folder = r"../0-Data/1-library-pinyin"
target_folder = r"../0-Data/2-blobs"
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

blobs = pd.read_csv('all_blobs.csv')

def process_row(row):
    print(row)
    # Read the particle coordinates
    x = row['image_x']
    y = row['image_y']
    w = row['image_w']
    h = row['image_h']

    # Read the collage image
    collage_path = os.path.join(source_folder, row['collage_file'])
    collage_image = cv2.imread(collage_path)
    if collage_image is None:
        print(f"Failed to read image from {collage_path}")
        return

    # Check if the coordinates are within the image boundaries
    if x < 0 or y < 0 or x + w > collage_image.shape[1] or y + h > collage_image.shape[0]:
        print(f"Invalid coordinates for image {collage_path}: x={x}, y={y}, w={w}, h={h}")
        return

    # Crop the particle from the image
    blob = collage_image[y:y + h, x:x + w]

    # Create the subfolder
    subfolder_path = os.path.join(target_folder, row['set'], row['species_name'])
    try:
        os.makedirs(subfolder_path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {subfolder_path}: {e}")
        return

    # Save the particle image; name the image as species name + id
    new_image_path = os.path.join(subfolder_path, row['image_id'] + '.png')
    if not cv2.imwrite(new_image_path, blob, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
        print(f"Failed to save image to {new_image_path}")
        return

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor() as executor:
    executor.map(process_row, [row for _, row in blobs.iterrows()])