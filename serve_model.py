import os
import argparse
import numpy as np
import pandas as pd
import mlflow
from sklearn.datasets import load_diabetes, load_breast_cancer

def load_sample_data(dataset_name):
    """
    Load sample data for prediction.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('diabetes' or 'breast_cancer')
    
    Returns:
    --------
    X_sample : numpy array
        Sample features for prediction
    """
    if dataset_name == 'diabetes':
        dataset = load_diabetes()
    elif dataset_name == 'breast_cancer':
        dataset = load_breast_cancer()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Get 5 random samples
    indices = np.random.choice(len(dataset.data), size=5, replace=False)
    X_sample = dataset.data[indices]
    
    return X_sample, dataset.feature_names

def serve_local_model(run_id, dataset_name):
    """
    Load and serve a model locally from MLflow.
    
    Parameters:
    -----------
    run_id : str
        MLflow run ID for the model to load
    dataset_name : str
        Name of the dataset to use for sample predictions
    """
    try:
        # Load the model
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model from run {run_id}")
        
        # Load sample data
        X_sample, feature_names = load_sample_data(dataset_name)
        
        # Make predictions
        predictions = model.predict(X_sample)
        
        # Print predictions
        print("\nSample Predictions:")
        for i, pred in enumerate(predictions):
            print(f"Sample {i+1}: {pred}")
        
        # Show feature importance if available
        underlying_model = model.unwrap_python_model()
        
        if hasattr(underlying_model, 'feature_importances_'):
            importances = underlying_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importances:")
            print(importance_df.head(10))
    
    except Exception as e:
        print(f"Error serving model: {e}")

def main():
    parser = argparse.ArgumentParser(description='Serve MLflow models for prediction')
    parser.add_argument('--run-id', type=str, required=True, help='MLflow run ID of the model to serve')
    parser.add_argument('--dataset', type=str, default='breast_cancer', 
                        choices=['breast_cancer', 'diabetes'], 
                        help='Dataset to use for sample predictions')
    args = parser.parse_args()
    
    serve_local_model(args.run_id, args.dataset)

if __name__ == "__main__":
    main()
