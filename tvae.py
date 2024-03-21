import pandas as pd
from sdv.tabular import TVAE

def generate_synthetic_data_with_tvae(csv_file, num_rows):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Initialize the TVAE model
    model = TVAE()

    # Fit the model on the data
    model.fit(data)

    # Generate synthetic data
    synthetic_data = model.sample(num_rows)

    return synthetic_data

# Example usage
csv_file = 'your_data.csv'  # Replace with the path to your CSV file
num_rows = 100  # Number of rows of synthetic data to generate
synthetic_data = generate_synthetic_data_with_tvae(csv_file, num_rows)
print(synthetic_data)
