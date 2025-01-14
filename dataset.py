#/bin/python3

import yaml
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


# Load configuration from config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def download_csv(url, output_file):
    """
    Downloads a CSV file from the web and stores up to 200 rows into another file.
    """
    try:
        # Access the webpage and download the CSV content
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        # Read the CSV content
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        # Limit to 200 rows
        df = df.head(200)
        
        # Save the processed dataset
        df.to_csv(output_file, index=False)
        print(f"Data successfully downloaded and saved to {output_file}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the CSV file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def analyze_dataset(file_path):
    """
    Opens the dataset and prints the number of rows, columns, and unique classes in the last column.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Analyze its structure
        rows, cols = df.shape
        last_column_classes = df.iloc[:, -1].nunique()
        
        print(f"Number of rows: {rows}")
        print(f"Number of columns: {cols}")
        print(f"Number of unique classes in the last column: {last_column_classes}")
        
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while analyzing the dataset: {e}")

def predict_with_linear_regression(df, target_column):
    """
    Performs linear regression on the dataset to predict the target column.
    """
    try:
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        return model
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

# Main Execution
if __name__ == "__main__":
    # Dataset URL (replace with the actual Kaggle dataset URL or local file)
    dataset_url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"  # Example CSV file
    output_file = "For_Prediction.csv"
    
    # Step 1: Download and process the dataset
    download_csv(dataset_url, output_file)
    
    # Step 2: Analyze the dataset
    df = analyze_dataset(output_file)
    
    # Step 3: Perform prediction (assuming the last column is the target variable)
    if df is not None:
        target_column = df.columns[-1]
        predict_with_linear_regression(df, target_column)

