from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers, optimizers, metrics
from sklearn.model_selection import train_test_split
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

def generate_X(data: Dict) -> List:
    """
    """

    ids = []
    tensors = []
    
    for id, method in data.items():
        ids.append(id)
        tensors.append(np.array(image.img_to_array(method())))

    return ids, tensors

def generate_y(metadata, ids):

    df = metadata[['file', 'health']]
    df['file'] = pd.Categorical(df['file'], categories=ids, ordered=True)
    df.sort_values(by='file')

    y_keys = {
        "healthy":np.array([1, 0, 0, 0, 0, 0]),
        "few varrao, hive beetles":np.array([0, 1, 0, 0, 0, 0]),
        "Varroa, Small Hive Beetles":np.array([0, 0, 1, 0, 0, 0]),
        "ant problems":np.array([0, 0, 0, 1, 0, 0]),
        "hive being robbed":np.array([0, 0, 0, 0, 1, 0]),
        "missing queen":np.array([0, 0, 0, 0, 0, 1])
    }

    y = [y_keys[i] for i in df['health']]
    
    return [y]

def split_data(X, y) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=13
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    model = models.Sequential([
        layers.Convolution2D(11, (3, 3), input_shape=(64, 64, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='SAME'),
        layers.Convolution2D(21, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding="SAME"),
        # third convo layer with more feature filter size, 41 for better detection.
        layers.Convolution2D(41, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding="SAME"),
        # flattening to input the fully connected layers
        layers.Flatten(),
        layers.Dense(200, activation='relu'),
        layers.Dropout(.2),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy", metrics.Precision(), metrics.Recall()]
    )

    model.fit(
        np.array(X_train),
        np.array(y_train),
    #    validation_data=(np.array(X_test), np.array(y_test)),
        verbose=True,
        shuffle=True,
        epochs=5
    )

    return model

def evaluate_model(model: TensorFlowModelDataset, X_test, y_test):
    """Evaluate the model into test dataset
        Args:
            model: Trained model.
            X_test: Testing data of independent features.
            y_test: labels of the testing set
    """
    print(".........Evaluation Started.........")
    history = model.evaluate(np.array(X_test), np.array(y_test))