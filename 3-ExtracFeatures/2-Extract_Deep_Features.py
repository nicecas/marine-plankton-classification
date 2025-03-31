import pandas as pd
import os
import glob
from PIL import Image
from torchvision.transforms import v2
import torch

from torchvision.models.feature_extraction import create_feature_extractor


def feature_extractor(model_path):
    """
    Load the model from the given path and create a feature extractor that
    outputs the features from the "flatten" layer.
    """
    model = torch.load(model_path, weights_only=False).cuda()
    features = ['flatten']
    feature_extractor = create_feature_extractor(model, return_nodes=features)
    return feature_extractor


# Define data transformations for training and validation
data_transforms = {
    'train': v2.Compose([
        v2.PILToTensor(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        v2.ToDtype(torch.float32, scale=True),  # Convert image to float32 and scale pixel values to [0, 1]
        v2.Normalize(mean=(0.704, 0.740, 0.781), std=(0.115, 0.135, 0.160))
    ]),
    'val': v2.Compose([
        v2.PILToTensor(),
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.704, 0.740, 0.781), std=(0.115, 0.135, 0.160))
    ]),
}


def find_png_paths(all_image_paths, image_name):
    """
    Filter and return the paths that match the specified image name.

    Args:
        all_image_paths (list): List of paths for all PNG images.
        image_name (str): The image name to be found (without extension).

    Returns:
        list: List of matching image paths.
    """
    selected_paths = [file for file in all_image_paths if os.path.splitext(os.path.basename(file))[0] == image_name]
    return selected_paths


def extrac_features(FE, image_path):
    """
    Extract features for a single image using the feature extractor.

    Args:
        FE (callable): The feature extractor.
        image_path (str): The path to the image file.

    Returns:
        dict: Dictionary containing the extracted features.
    """
    image = Image.open(image_path)
    input = data_transforms['val'](image).cuda()
    input = input.unsqueeze(0)  # Add batch dimension
    features = FE(input)
    return features


def main():
    # Check and create the features folder if it does not exist
    features_folder = "./features"
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
        print(f"Created folder: {features_folder}")

    # Read metadata from CSV file
    meta_csv_path = r'../1-DataPreparation/all_blobs.csv'
    meta_df = pd.read_csv(meta_csv_path)

    # Get all blob images (PNG files)
    png_files = glob.glob(os.path.join(r'../0-Data/2-blobs', '**', '*.png'), recursive=True)

    # Set model paths
    # Since the SGD optimizer performs well in this scenario, we select the model
    # trained using the SGD optimizer for feature extraction.
    model_paths = glob.glob(r'../2-TrainingDNN/models/*SGD*best.pt')

    for model_path in model_paths:
        print(model_path)
        FE = feature_extractor(model_path)
        all_image_features = pd.DataFrame()  # DataFrame to store extracted features for each image

        # Iterate over each row in the metadata dataframe
        for _, row in meta_df.iterrows():
            image_id = row['image_id']
            # Get the first matching image path for the current image id
            image_path = find_png_paths(png_files, image_id)[0]
            features = extrac_features(FE, image_path)
            # Convert extracted features to a DataFrame
            features = pd.DataFrame(features['flatten'].cpu().detach().numpy())
            # Add additional information from metadata
            features['species_name'] = row['species_name']
            features['image_id'] = row['image_id']
            features['set'] = row['set']

            # Concatenate current image features to the overall DataFrame
            all_image_features = pd.concat([all_image_features, features], axis=0)

        print(all_image_features.head())

        # Save the extracted features to a CSV file
        model_name = os.path.basename(model_path).split('_')[0]
        output_csv_path = os.path.join(features_folder, model_name + ".csv")
        all_image_features.to_csv(output_csv_path, header=True, index=False)
        print(f"Saved features to: {output_csv_path}")


if __name__ == "__main__":
    main()