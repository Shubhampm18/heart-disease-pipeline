import json
import os
import sys

import joblib
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")


def load_bundle():
    return joblib.load(MODEL_PATH)


def predict(input_dict: dict) -> str:
    bundle = load_bundle()
    features = bundle["feature_columns"]

    missing = [c for c in features if c not in input_dict]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

    row = {c: input_dict[c] for c in features}
    df = pd.DataFrame([row])
    prediction = bundle["pipeline"].predict(df)[0]
    return "disease" if int(prediction) == 1 else "no_disease"


if __name__ == "__main__":
    data = json.load(sys.stdin)
    print(predict(data))