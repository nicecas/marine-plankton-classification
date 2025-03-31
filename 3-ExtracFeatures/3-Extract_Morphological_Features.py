import pandas as pd

# Load the CSV file
file_path = r'../1-DataPreparation/all_blobs.csv'
df = pd.read_csv(file_path)

df.head()

# Extract columns from 'raw_area' to 'set'
start_col = 'raw_area'
end_col = 'set'

# Get the index positions of the columns
start_idx = df.columns.get_loc(start_col)
end_idx = df.columns.get_loc(end_col)

# Extract the specified range of columns
subset_df = df.iloc[:, start_idx:end_idx+1]

# Remove columns that are completely zero
subset_df = subset_df.loc[:, (subset_df != 0).any(axis=0)]

subset_df.head()

# Save the subset to a new CSV file

subset_df.to_csv("./features/morphological.csv", header=True, index=False)