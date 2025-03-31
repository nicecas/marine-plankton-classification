import os
import glob
import pandas as pd

def load_csv_data(csv_paths):
    """
    Load feature data from CSV files.

    The feature extractor names are derived from the CSV file names. This function
    returns a list of feature extractor names and a corresponding list of DataFrames.
    """
    feature_extractors = []  # List to store feature extractor names (derived from file names)
    features = []            # List to store corresponding DataFrames for each CSV file

    for csv_path in csv_paths:
        # Extract file name without extension to serve as feature extractor name
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        feature_extractors.append(file_name)
        features.append(df)

    return feature_extractors, features

def check_features_imageid_same(features):
    """
    Verify that every feature DataFrame shares the same 'image_id' column.

    Returns True if all DataFrames have identical 'image_id' values; otherwise, returns False.
    """
    image_ids_list = [df['image_id'] for df in features]
    first_image_ids = image_ids_list[0]

    for other_image_ids in image_ids_list[1:]:
        if not first_image_ids.equals(other_image_ids):
            return False

    return True

def get_fused_features(feature_extractors, features, fusions):
    """
    Generate fused features according to the fusion scheme.

    For a single integer entry in 'fusions', the corresponding original feature is retained.
    For list entries, the specified features (by index) are concatenated (fused). Only the columns
    with float values are fused, and the non-fused columns 'species_name' and 'set' are appended
    from the first DataFrame in the list.
    """
    fused_feature_extractors = []  # List to store names for the fused feature groups
    fused_features = []            # List to store fused DataFrames

    for fusion in fusions:
        if isinstance(fusion, int):
            # Single feature: keep the original feature without fusion
            fused_feature_extractors.append(feature_extractors[fusion])
            fused_features.append(features[fusion])
        else:
            # Fusion: fuse the specified features listed in the fusion scheme
            dataframes_to_fuse = [features[i] for i in fusion]
            # Concatenate only columns with float type across the specified DataFrames
            fused_df = pd.concat([df.select_dtypes(include='float') for df in dataframes_to_fuse], axis=1)
            # Append additional columns (e.g., species_name and set) from the first DataFrame
            fused_df = pd.concat([fused_df, dataframes_to_fuse[0][['species_name', 'set']]], axis=1)
            # Create a fusion key by joining the feature extractor names that were fused
            fusion_key = '_'.join([feature_extractors[i] for i in fusion])
            fused_feature_extractors.append(fusion_key)
            fused_features.append(fused_df)

    return fused_feature_extractors, fused_features

def generate_fusions(feature_extractors):
    """
    Dynamically generate the fusion schemes.

    First, this function includes each feature individually. Then, if there is any
    morphological feature identified (based on the keyword "morph" in its name, case insensitive),
    it creates fusion pairs between the morphological feature and each other feature.
    """
    all_indices = list(range(len(feature_extractors)))
    fusions = []

    # Include each feature individually
    for idx in all_indices:
        fusions.append(idx)

    # Identify the index of the morphological feature (if any)
    morphological_indices = [
        idx for idx, name in enumerate(feature_extractors) if "morph" in name.lower()
    ]

    # If a morphological feature exists, fuse every other feature with it
    if morphological_indices:
        morph_idx = morphological_indices[0]  # Using the first detected morphological feature
        for idx in all_indices:
            if idx != morph_idx:
                fusions.append([idx, morph_idx])

    return fusions

if __name__ == "__main__":
    # Define the directory containing the feature CSV files
    folder_path = '../3-ExtracFeatures/features'
    csv_paths = glob.glob(os.path.join(folder_path, '*.csv'))

    # Load the CSV data and retrieve feature extractor names and DataFrames
    feature_extractors, features = load_csv_data(csv_paths)
    print("Original feature extractors:", feature_extractors)

    # Ensure that all DataFrames have identical 'image_id' column values
    if not check_features_imageid_same(features):
        print("Error: The 'image_id' columns do not match across all feature files.")
        exit(1)

    # Dynamically generate fusion schemes
    fusions = generate_fusions(feature_extractors)
    print("Fusion schemes:", fusions)

    # Create fused features based on the generated fusion schemes
    fused_feature_extractors, fused_features = get_fused_features(feature_extractors, features, fusions)
    print("Fused feature extractors:", fused_feature_extractors)

    # Define the output directory for fused feature CSV files
    output_folder = './FusedFeatures'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each fused DataFrame to CSV files
    for fe_name, df in zip(fused_feature_extractors, fused_features):
        output_path = os.path.join(output_folder, fe_name + '.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved fused feature file: {output_path}")