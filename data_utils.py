import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(dataset_name, test_size=0.2, random_state=42):
    """
    Load and preprocess a dataset for machine learning tasks.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load ('diabetes' for regression, 'breast_cancer' for classification)
    test_size : float, default=0.2
        Proportion of the dataset to be used as test set
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy arrays
        Split and preprocessed data
    """
    if dataset_name == 'diabetes':
        # Regression dataset
        dataset = load_diabetes()
        X, y = dataset.data, dataset.target
        # Reshape target for TensorFlow
        y = y.reshape(-1, 1)
        
    elif dataset_name == 'breast_cancer':
        # Classification dataset
        dataset = load_breast_cancer()
        X, y = dataset.data, dataset.target
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose 'diabetes' or 'breast_cancer'.")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Loaded {dataset_name} dataset:")
    print(f"- Training set: {X_train.shape[0]} samples")
    print(f"- Test set: {X_test.shape[0]} samples")
    print(f"- Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test
