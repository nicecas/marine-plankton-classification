import os
import pandas as pd
from sklearn.metrics import classification_report
import glob


def load_data(base_path):
    data = {}
    # Use glob to get paths of all tune.csv files
    csv_files = glob.glob(os.path.join(base_path, '**', '*tune.csv'), recursive=True)

    for csv_file in csv_files:
        # Retrieve the file name and the name of the parent directory
        file_name = os.path.basename(csv_file)
        classifier = file_name[:-9]  # Remove the '_tune.csv' suffix
        extractor = os.path.basename(os.path.dirname(csv_file))

        if extractor not in data:
            data[extractor] = {}

        df = pd.read_csv(csv_file)
        data[extractor][classifier] = df[['species_name', 'prediction_label']]

    return data


def evaluate_models(data):
    all_results = []
    for extractor, classifiers in data.items():
        for classifier, df in classifiers.items():
            y_true = df['species_name']
            y_pred = df['prediction_label']
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            result = {
                'FeatureExtractor': extractor,
                'Classifier': classifier,
            }
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        result[f'{label}_{metric}'] = value
                else:
                    result[label] = metrics
            all_results.append(result)
    return pd.DataFrame(all_results)


# Load data
data = load_data('./models')

# Evaluate models
results_df = evaluate_models(data)

# Save results
results_df.to_csv('3-val_metrics.csv', index=False)