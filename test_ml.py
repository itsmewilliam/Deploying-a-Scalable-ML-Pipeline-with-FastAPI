import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

# Load some data to use for all tests
sample_data = pd.read_csv("data/census.csv").sample(n=500, random_state=1)

# These are the categorical features used in this project
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the data and train a model so we can test things
X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
model = train_model(X, y)


# First test
def test_model_is_random_forest():
    """
    Make sure the model is a RandomForestClassifier.
    """
    assert isinstance(model, RandomForestClassifier)


# Second test
def test_inference_outputs_predictions():
    """
    Test that inference returns predictions for all input rows.
    """
    preds = inference(model, X)
    assert len(preds) == len(X)


# Third test
def test_metrics_are_in_valid_range():
    """
    Check that precision, recall, and f1 are all between 0 and 1.
    """
    preds = inference(model, X)
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
