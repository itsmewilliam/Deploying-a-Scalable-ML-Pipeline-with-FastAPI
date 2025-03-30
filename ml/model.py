import pickle
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)  # Fit the model to the training data
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)  # Use the model to predict values
    return preds

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, "wb") as file:  # Open the file in write-binary mode
        pickle.dump(model, file)    # Save the model using pickle

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    with open(path, "rb") as file:  # Open the file in read-binary mode
        model = pickle.load(file)   # Load the model
    return model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data.

    For example: how well the model performs when 'education' is 'Bachelors'.
    """

    # Filter the data to only rows where the column matches the slice value
    data_slice = data[data[column_name] == slice_value]

    # Process the slice data (we're not training here)
    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run predictions on the slice
    preds = inference(model, X_slice)

    # Calculate metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)

    return precision, recall, fbeta
