# MLflow Hyperparameter Tuning Guide

A comprehensive guide for using MLflow to track and tune hyperparameters for both TensorFlow and scikit-learn models.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Step 1: Environment Setup](#step-1-environment-setup)
- [Step 2: Understanding the Configuration File](#step-2-understanding-the-configuration-file)
- [Step 3: Data Preparation](#step-3-data-preparation)
- [Step 4: Model Creation](#step-4-model-creation)
- [Step 5: MLflow Tracking](#step-5-mlflow-tracking)
- [Step 6: Hyperparameter Tuning](#step-6-hyperparameter-tuning)
- [Step 7: Running Experiments](#step-7-running-experiments)
- [Step 8: Accessing MLflow UI](#step-8-accessing-mlflow-ui)
- [Step 9: Model Serving](#step-9-model-serving)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

This project demonstrates how to use MLflow for hyperparameter tuning with both TensorFlow and scikit-learn models. The entire pipeline is driven by a YAML configuration file, making it easy to experiment with different models and settings.

Key features:
- Configuration-driven experiments
- Support for both TensorFlow and scikit-learn models
- Automatic tracking of hyperparameters, metrics, and models
- Hyperparameter grid search
- SQLite backend for persistent tracking
- Model serving capabilities

## Project Structure

```
MLFlow-simple-train/
├── config.yaml            # Configuration file
├── train.py               # Main training script
├── data_utils.py          # Data loading and preprocessing
├── model_utils.py         # Model creation utilities
├── mlflow_utils.py        # MLflow tracking utilities
├── serve_model.py         # Model serving script
└── README.md              # This file
```

## Step 1: Environment Setup

### Create a Virtual Environment

```bash
# Create a new directory for your project
mkdir MLFlow-simple-train
cd MLFlow-simple-train

# Create a virtual environment
python -m venv mlflow_env

# Activate the environment
# On Windows:
# mlflow_env\Scripts\activate
# On macOS/Linux:
source mlflow_env/bin/activate
```

### Install Required Packages

```bash
# Install all necessary packages
pip install mlflow tensorflow scikit-learn pandas numpy pyyaml matplotlib seaborn
```

## Step 2: Understanding the Configuration File

The `config.yaml` file controls all aspects of the experiment:

```yaml
# General configuration
seed: 42
dataset: "breast_cancer"  # "breast_cancer" for classification or "diabetes" for regression
task: "classification"    # "classification" or "regression"
test_size: 0.2
model_type: "sklearn"     # "sklearn" or "tensorflow"
optimization_metric: "accuracy"  # Metric to optimize
optimization_mode: "max"  # "max" to maximize, "min" to minimize

# MLflow configuration
mlflow:
  experiment_name: "hyperparameter_tuning_demo"
  tracking_uri: "sqlite:///mlruns/mlflow.db"  # Use SQLite backend
  autologging: true  # Enable autologging

# Scikit-learn model configuration
sklearn:
  model_type: "random_forest"  # "random_forest", "logistic_regression", "svm"
  hyperparameters:
    n_estimators: [50, 100, 200]
    max_depth: [10, 20, 30, null]
    min_samples_split: [2, 5, 10]
    random_state: [42]

# TensorFlow model configuration
tensorflow:
  hyperparameters:
    units: [32, 64, 128]
    num_layers: [1, 2, 3]
    learning_rate: [0.001, 0.01]
    dropout_rate: [0.0, 0.2, 0.5]
    activation: ["relu"]
    batch_size: [32, 64]
    epochs: [20]
    patience: [5]
    random_state: [42]
```

Key configuration sections:
- **General settings**: Control the seed, dataset, and task type
- **MLflow settings**: Configure experiment name and tracking backend
- **Model-specific settings**: Define hyperparameters to tune for each model type

## Step 3: Data Preparation

The `data_utils.py` module handles data loading and preprocessing:

```python
# Example from data_utils.py
def load_dataset(dataset_name, test_size=0.2, random_state=42):
    """Load and preprocess a dataset for machine learning tasks."""
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
        
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state)
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
```

This function:
1. Loads either a classification (breast_cancer) or regression (diabetes) dataset
2. Splits the data into training and test sets
3. Normalizes features using StandardScaler
4. Returns processed data ready for model training

## Step 4: Model Creation

The `model_utils.py` module creates models based on configuration:

### Creating a scikit-learn Model

```python
# Example from model_utils.py
def create_sklearn_model(model_type, params, task='classification'):
    """Create a scikit-learn model based on specified type and parameters."""
    if task == 'classification':
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                random_state=params.get('random_state', 42)
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                C=params.get('C', 1.0),
                penalty=params.get('penalty', 'l2'),
                random_state=params.get('random_state', 42)
            )
        # More models...
    
    elif task == 'regression':
        # Regression models...
    
    return model
```

### Creating a TensorFlow Model

```python
# Example from model_utils.py
def create_tensorflow_model(params, input_shape, output_shape, task='classification'):
    """Create a TensorFlow model based on specified parameters."""
    # Set random seed for reproducibility
    tf.random.set_seed(params.get('random_state', 42))
    
    # Create model
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    
    # Hidden layers
    for i in range(params.get('num_layers', 2)):
        model.add(tf.keras.layers.Dense(
            units=params.get('units', 64),
            activation=params.get('activation', 'relu')
        ))
        
        # Add dropout if specified
        if params.get('dropout_rate', 0.0) > 0:
            model.add(tf.keras.layers.Dropout(rate=params.get('dropout_rate')))
    
    # Output layer (classification or regression)
    if task == 'classification':
        if output_shape == 1:  # Binary classification
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # Multi-class classification
            model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
    else:  # Regression
        model.add(tf.keras.layers.Dense(output_shape))
        loss = 'mse'
        metrics = ['mae']
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
        loss=loss,
        metrics=metrics
    )
    
    return model
```

## Step 5: MLflow Tracking

The `mlflow_utils.py` module handles all MLflow tracking functionality:

### Setting Up MLflow

```python
def setup_mlflow(config):
    """Set up MLflow tracking."""
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
```

### Tracking scikit-learn Models

```python
def log_sklearn_model(model, X_train, X_test, y_train, y_test, params, config):
    """Train and log a scikit-learn model with MLflow."""
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
            accuracy = accuracy_score(y_test, y_pred)
            metrics['accuracy'] = accuracy
            mlflow.log_metric("accuracy", accuracy)
        elif task == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics.update({'mse': mse, 'mae': mae, 'r2': r2})
            mlflow.log_metrics(metrics)
        
        # Log the model
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=mlflow.models.infer_signature(X_test, y_pred)
        )
```

### Tracking TensorFlow Models

```python
def log_tensorflow_model(params, X_train, X_test, y_train, y_test, config):
    """Train and log a TensorFlow model with MLflow."""
    # MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Enable autologging (optional)
        if config.get('mlflow', {}).get('autologging', False):
            mlflow.tensorflow.autolog()
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 20),
            batch_size=params.get('batch_size', 32),
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log the model
        mlflow.tensorflow.log_model(
            model, 
            "model",
            signature=mlflow.models.infer_signature(X_test, y_pred)
        )
```

## Step 6: Hyperparameter Tuning

The hyperparameter tuning logic in `train.py` generates and evaluates all combinations:

```python
def generate_hyperparameter_grid(hyperparams):
    """Generate all combinations of hyperparameters."""
    keys = hyperparams.keys()
    values = hyperparams.values()
    combinations = list(product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]

def run_hyperparameter_tuning(config, X_train, X_test, y_train, y_test):
    """Run hyperparameter tuning with MLflow tracking."""
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
    
    # Run trials
    for i, params in enumerate(param_combinations):
        print(f"\nTrial {i+1}/{len(param_combinations)}")
        
        if config.get('model_type') == 'sklearn':
            model = create_sklearn_model(model_type, params, task=config.get('task'))
            log_sklearn_model(model, X_train, X_test, y_train, y_test, params, config)
        else:  # tensorflow
            log_tensorflow_model(params, X_train, X_test, y_train, y_test, config)
```

## Step 7: Running Experiments

To run an experiment:

```bash
# Activate your virtual environment
source mlflow_env/bin/activate

# Run the training script
python train.py

# Or with a custom config file:
# python train.py --config custom_config.yaml
```

The `train.py` script:
1. Loads configuration from `config.yaml`
2. Sets up MLflow tracking
3. Loads and preprocesses the dataset
4. Runs hyperparameter tuning
5. Finds the best model based on the optimization metric

## Step 8: Accessing MLflow UI

To view your experiment results:

```bash
# Start the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open your browser to http://127.0.0.1:5000

In the MLflow UI, you can:
- View all experiments
- Compare runs side-by-side
- Sort and filter runs based on parameters and metrics
- View detailed information for each run
- Download logged artifacts

## Step 9: Model Serving

To use a trained model for inference:

```bash
# Serve a model with its run ID
python serve_model.py --run-id RUN_ID --dataset breast_cancer
```

The `serve_model.py` script:
1. Loads the model from MLflow
2. Prepares sample data for prediction
3. Makes predictions with the loaded model
4. Displays the results and feature importance (if available)

## Best Practices

1. **Reproducibility**: Always set random seeds in all components for consistent results.

2. **SQLite Backend**: Use SQLite for persistence between sessions:
   ```python
   mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
   ```

3. **Autologging**: Enable autologging for automatic parameter and metric tracking:
   ```python
   mlflow.sklearn.autolog()  # For scikit-learn
   mlflow.tensorflow.autolog()  # For TensorFlow
   ```

4. **Model Signatures**: Always include model signatures for better serving:
   ```python
   mlflow.sklearn.log_model(
       model, "model",
       signature=mlflow.models.infer_signature(X_test, y_pred)
   )
   ```

5. **Experiment Organization**: Use meaningful experiment names to organize related runs.

6. **Hyperparameter Ranges**: Start with broad ranges and then refine based on results.

## Troubleshooting

**Issue**: MLflow UI shows no experiments
- Check the tracking URI is set correctly
- Ensure the SQLite database path exists
- Verify you've run at least one experiment

**Issue**: Model training fails
- Check data preprocessing steps
- Verify model hyperparameters are valid
- Ensure you have enough memory for model training

**Issue**: Cannot load saved model
- Verify the run ID exists
- Check the model was logged successfully
- Ensure dependencies match between training and serving environments

**Issue**: Slow hyperparameter tuning
- Reduce the number of hyperparameter combinations
- Use smaller datasets for initial exploration
- Consider using more efficient search strategies 

**Issue**: `WARNING:root:Malformed experiment 'X'. Detailed error Yaml file './mlruns/X/meta.yaml' does not exist.`
- This error occurs when MLflow finds a corrupted or incomplete experiment directory
- Solutions:
  1. Delete the corrupted experiment folder:
     ```bash
     # Replace X with the experiment ID from the error message
     rm -rf ./mlruns/X
     ```
  2. Create a new tracking URI to avoid the corrupted data:
     ```bash
     # In config.yaml, update the tracking_uri
     mlflow:
       tracking_uri: "sqlite:///mlruns/new_mlflow.db"
     ```
  3. If you need to recover experiments, manually create the missing meta.yaml:
     ```bash
     # Create directory if it doesn't exist
     mkdir -p ./mlruns/X
     
     # Create a basic meta.yaml file
     echo "name: recovered_experiment
artifact_location: ./mlruns/X
lifecycle_stage: active
creation_time: $(date +%s)000" > ./mlruns/X/meta.yaml
     ```

---

For more information, visit the [MLflow documentation](https://mlflow.org/docs/latest/index.html).
