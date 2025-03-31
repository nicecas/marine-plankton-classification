import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split

'''
This script is used to convert the metadata files of the FlowCam classification dataset 
(.FlowCam classification dataset preprocessing) into CSV format for easy subsequent usage.
1. The flb file can be read as text.
2. The first few lines are variable names, and the remaining lines are the values for each variable.
'''

flb_folder = r"../0-Data/1-library-pinyin"
flb_paths = glob.glob(os.path.join(flb_folder, '*.flb'))

# Create an empty DataFrame to store all the data
data = pd.DataFrame()

for flb_path in flb_paths:
    print(flb_path)
    with open(flb_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        # From the third to the 67th line, each line contains the name of a variable
        var_names = [line.split('|')[0] for line in lines[2:67]]
        # Read data starting from the 67th line onward
        df = pd.read_csv(flb_path, sep="|", header=66, dtype=str, names=var_names, encoding='utf-8')
        data = pd.concat([data, df], ignore_index=True)

# Remove duplicates (when creating the classification dataset, there might be manual errors where the same image is imported twice)
data = data.drop_duplicates(subset='image_id', keep='first')


'''
Extract species names and partition the dataset
'''
# Extract species names
data['species_name'] = data['collage_file'].apply(lambda x: x.split('_')[0])


data_partitions = []

# Group by the 'species_name' column
for _, group in data.groupby('species_name'):
    # Split into training (70%) and temporary (30%) sets
    train, temp = train_test_split(group, test_size=0.3, random_state=42)
    # Split the temporary set evenly into validation (val) and test sets
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Add a column to each subset to indicate its partition
    train['set'] = 'train'
    val['set'] = 'val'
    test['set'] = 'test'

    # Add each subset to the data_partitions list
    data_partitions.extend([train, val, test])

# Concatenate all subsets back into a single DataFrame
result_df = pd.concat(data_partitions)

result_df.to_csv('all_blobs.csv', index=False)