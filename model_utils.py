import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

def create_sklearn_model(model_type, params, task='classification'):
    """
    Create a scikit-learn model based on the specified type and parameters.
    
    Parameters:
    -----------
    model_type : str
        Type of sklearn model ('random_forest', 'logistic_regression', 'svm')
    params : dict
        Model hyperparameters
    task : str, default='classification'
        'classification' or 'regression'
    
    Returns:
    --------
    model : scikit-learn model instance
    """
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
        elif model_type == 'svm':
            model = SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                probability=True,
                random_state=params.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported sklearn classification model: {model_type}")
    
    elif task == 'regression':
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                random_state=params.get('random_state', 42)
            )
        elif model_type == 'ridge':
            model = Ridge(
                alpha=params.get('alpha', 1.0),
                random_state=params.get('random_state', 42)
            )
        elif model_type == 'svm':
            model = SVR(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf')
            )
        else:
            raise ValueError(f"Unsupported sklearn regression model: {model_type}")
    
    else:
        raise ValueError(f"Unsupported task: {task}. Choose 'classification' or 'regression'.")
    
    return model

def create_tensorflow_model(params, input_shape, output_shape, task='classification'):
    """
    Create a TensorFlow model based on the specified parameters.
    
    Parameters:
    -----------
    params : dict
        Model hyperparameters
    input_shape : int
        Number of input features
    output_shape : int
        Number of output classes or 1 for regression
    task : str, default='classification'
        'classification' or 'regression'
    
    Returns:
    --------
    model : compiled TensorFlow model
    """
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
    
    # Output layer
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
