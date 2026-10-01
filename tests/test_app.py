import joblib
import pytest

from predict import MODEL_PATH, predict


def valid_sample():
    return {
        "age": 55,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 7,
    }


def test_saved_model_can_be_loaded():
    bundle = joblib.load(MODEL_PATH)
    assert "pipeline" in bundle
    assert "feature_columns" in bundle


def test_valid_sample_output_shape_and_type():
    prediction = predict(valid_sample())
    assert isinstance(prediction, str)
    assert prediction in {"disease", "no_disease"}


def test_missing_feature_rejected():
    bad = valid_sample()
    del bad["chol"]

    with pytest.raises(ValueError, match="Missing required feature"):
        predict(bad)