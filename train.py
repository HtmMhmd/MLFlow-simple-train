import os
import sys
import argparse
import yaml
import numpy as np
import mlflow
import random
import tensorflow as tf
from itertools import product
from data_utils import load_dataset
from model_utils import create_sklearn_model, create_tensorflow_model
from mlflow_utils import (
    load_config, setup_mlflow, log_sklearn_model, 
    log_tensorflow_model, find_best_run
)

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def generate_hyperparameter_grid(hyperparams):
    """
    Generate all combinations of hyperparameters.
    
    Parameters:
    -----------
    hyperparams : dict
        Dictionary of hyperparameters with lists of values
    
    Returns:
    --------
    list of dicts
        All possible combinations of hyperparameters
    """
    keys = hyperparams.keys()
    values = hyperparams.values()
    combinations = list(product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]

def run_hyperparameter_tuning(config, X_train, X_test, y_train, y_test):
    """
    Run hyperparameter tuning with MLflow tracking.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    X_train, X_test, y_train, y_test : numpy arrays
        Training and test data
    """
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Get hyperparameters to tune
    if config.get('model_type') == 'sklearn':
        model_config = config.get('sklearn', {})
        model_type = model_config.get('model_type')
        hyperparams = model_config.get('hyperparameters', {})
    else:  # tensorflow
        model_config = config.get('tensorflow', {})
        hyperparams = model_config.get('hyperparameters', {})
    
    # Generate hyperparameter combinations
    param_combinations = generate_hyperparameter_grid(hyperparams)
    
    print(f"Running hyperparameter tuning with {len(param_combinations)} combinations")
    
    # Run trials
    for i, params in enumerate(param_combinations):
        print(f"\nTrial {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        
        if config.get('model_type') == 'sklearn':
            model = create_sklearn_model(
                model_type, 
                params, 
                task=config.get('task', 'classification')
            )
            metrics = log_sklearn_model(model, X_train, X_test, y_train, y_test, params, config)
        else:  # tensorflow
            metrics = log_tensorflow_model(params, X_train, X_test, y_train, y_test, config)
    
    # Find best run
    metric_name = config.get('optimization_metric', 'accuracy')
    mode = config.get('optimization_mode', 'max')
    
    print("\nFinding best model...")
    best_run = find_best_run(
        config.get('mlflow', {}).get('experiment_name', 'default_experiment'),
        metric_name,
        mode
    )
    
    if best_run:
        print("\n=== Best Model ===")
        print(f"Run ID: {best_run['run_id']}")
        print(f"Metric '{metric_name}': {best_run[f'metrics.{metric_name}']}")
        print("Parameters:")
        for key in best_run.keys():
            if key.startswith('params.'):
                param_name = key.replace('params.', '')
                print(f"  {param_name}: {best_run[key]}")
    else:
        print("No best model found.")

def main():
    parser = argparse.ArgumentParser(description='Train models with MLflow tracking')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set up MLflow
        setup_mlflow(config)
        
        # Load dataset
        dataset_name = config.get('dataset', 'breast_cancer')
        X_train, X_test, y_train, y_test = load_dataset(
            dataset_name,
            test_size=config.get('test_size', 0.2),
            random_state=config.get('seed', 42)
        )
        
        # Run hyperparameter tuning
        run_hyperparameter_tuning(config, X_train, X_test, y_train, y_test)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
