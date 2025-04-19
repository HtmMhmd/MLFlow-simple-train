import os
import time
import yaml
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pyfunc
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file
    
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {e}")

def setup_mlflow(config):
    """
    Set up MLflow tracking.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with MLflow settings
    """
    # Set MLflow tracking URI if specified
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', None)
    if tracking_uri:
        if tracking_uri.startswith('sqlite:'):
            # Ensure the directory exists
            db_path = tracking_uri.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment name
    experiment_name = config.get('mlflow', {}).get('experiment_name', 'default_experiment')
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment name: {experiment_name}")

def log_sklearn_model(model, X_train, X_test, y_train, y_test, params, config):
    """
    Train and log a scikit-learn model with MLflow.
    
    Parameters:
    -----------
    model : scikit-learn model
        The model to train and log
    X_train, X_test, y_train, y_test : numpy arrays
        Training and test data
    params : dict
        Model hyperparameters
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    metrics : dict
        Model performance metrics
    """
    task = config.get('task', 'classification')
    
    with mlflow.start_run():
        start_time = time.time()
        
        # Log parameters
        mlflow.log_params(params)
        
        # Enable autologging (optional)
        if config.get('mlflow', {}).get('autologging', False):
            mlflow.sklearn.autolog()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate and log metrics
        metrics = {}
        if task == 'classification':
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            accuracy = accuracy_score(y_test, y_pred)
            metrics['accuracy'] = accuracy
            mlflow.log_metric("accuracy", accuracy)
            print(f"Accuracy: {accuracy:.4f}")
        elif task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics.update({'mse': mse, 'mae': mae, 'r2': r2})
            mlflow.log_metrics(metrics)
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Log training duration
        duration = time.time() - start_time
        mlflow.log_metric("training_duration", duration)
        metrics['training_duration'] = duration
        
        # Log the model
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=mlflow.models.infer_signature(X_test, y_pred)
        )
        
        # Log model type
        mlflow.set_tag("model_type", "sklearn")
        mlflow.set_tag("sklearn_model", config.get('sklearn', {}).get('model_type', 'unknown'))
        
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        
    return metrics

def log_tensorflow_model(params, X_train, X_test, y_train, y_test, config):
    """
    Train and log a TensorFlow model with MLflow.
    
    Parameters:
    -----------
    params : dict
        Model hyperparameters
    X_train, X_test, y_train, y_test : numpy arrays
        Training and test data
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    metrics : dict
        Model performance metrics
    """
    import tensorflow as tf
    from model_utils import create_tensorflow_model
    
    task = config.get('task', 'classification')
    
    # Determine output shape
    if task == 'classification':
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:  # One-hot encoded
            output_shape = y_train.shape[1]
        else:  # Binary or multi-class with integer labels
            output_shape = len(np.unique(y_train))
            if output_shape == 2:  # Binary
                output_shape = 1
    else:  # Regression
        output_shape = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    # Create model
    model = create_tensorflow_model(
        params, 
        X_train.shape[1], 
        output_shape, 
        task
    )
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=params.get('patience', 10),
            restore_best_weights=True
        )
    ]
    
    # MLflow run
    with mlflow.start_run():
        start_time = time.time()
        
        # Log parameters
        mlflow.log_params(params)
        
        # Enable autologging (optional)
        if config.get('mlflow', {}).get('autologging', False):
            mlflow.tensorflow.autolog()
        
        # Create MLflow callback for metric tracking
        class MLflowCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                for metric_name, metric_value in logs.items():
                    mlflow.log_metric(metric_name, metric_value, step=epoch)
        
        callbacks.append(MLflowCallback())
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 20),
            batch_size=params.get('batch_size', 32),
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        test_results = model.evaluate(X_test, y_test, verbose=0)
        metrics = {}
        
        # Log test metrics
        if task == 'classification':
            metrics['test_loss'] = test_results[0]
            metrics['test_accuracy'] = test_results[1]
            mlflow.log_metrics(metrics)
            print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        else:  # Regression
            metrics['test_loss'] = test_results[0]
            metrics['test_mae'] = test_results[1]
            mlflow.log_metrics(metrics)
            print(f"Test MAE: {metrics['test_mae']:.4f}")
        
        # Make predictions for signature
        y_pred = model.predict(X_test)
        
        # Log training duration
        duration = time.time() - start_time
        mlflow.log_metric("training_duration", duration)
        metrics['training_duration'] = duration
        
        # Log learning curves
        for metric_name, metric_values in history.history.items():
            for epoch, value in enumerate(metric_values):
                mlflow.log_metric(f"epoch_{metric_name}", value, step=epoch)
        
        # Log the model
        mlflow.tensorflow.log_model(
            model, 
            "model",
            signature=mlflow.models.infer_signature(X_test, y_pred)
        )
        
        # Log model type
        mlflow.set_tag("model_type", "tensorflow")
        
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        
    return metrics

def find_best_run(experiment_name, metric_name, mode='min'):
    """
    Find the best run for a given experiment based on a metric.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the MLflow experiment
    metric_name : str
        Name of the metric to optimize
    mode : str, default='min'
        'min' to minimize the metric, 'max' to maximize
    
    Returns:
    --------
    best_run : dict or None
        Information about the best run, or None if no runs found
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return None
    
    # Search for runs in the experiment
    query = f"experiment_id='{experiment.experiment_id}'"
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        print(f"No runs found for experiment '{experiment_name}'.")
        return None
    
    # Filter for runs that have the metric
    runs_with_metric = runs[runs[f"metrics.{metric_name}"].notnull()]
    
    if runs_with_metric.empty:
        print(f"No runs found with metric '{metric_name}'.")
        return None
    
    # Sort by the metric
    if mode == 'min':
        best_run = runs_with_metric.sort_values(f"metrics.{metric_name}").iloc[0]
    else:  # max
        best_run = runs_with_metric.sort_values(f"metrics.{metric_name}", ascending=False).iloc[0]
    
    return best_run.to_dict()
