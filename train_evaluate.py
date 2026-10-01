import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "heart_cleaned.csv"

TARGET = "target"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal",
]

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

MARGIN = 0.10
RANDOM_STATE = 42
ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL,
            ),
        ]
    )


if not DATA_PATH.exists():
    fail(f"Dataset not found at {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
except Exception as exc:
    fail(f"Could not load dataset: {exc}")

missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    fail(f"Missing required columns: {missing_cols}")

df = df.dropna(subset=FEATURES + [TARGET]).copy()
if df.empty:
    fail("No rows left after dropping missing values.")

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

if y.nunique() < 2:
    fail("Target has only one class after preprocessing.")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

subset = float(os.getenv("TRAIN_SUBSET_FRAC", "1.0"))
if subset < 1.0:
    X_train = X_train.sample(frac=subset, random_state=RANDOM_STATE)
    y_train = y_train.loc[X_train.index]

baseline = Pipeline([
    ("prep", make_preprocessor()),
    ("clf", DummyClassifier(strategy="most_frequent")),
])

if os.getenv("WEAK_MODEL", "0") == "1":
    candidate_clf = DummyClassifier(strategy="uniform", random_state=RANDOM_STATE)
    model_name = "DummyClassifier(strategy='uniform') [WEAK]"
else:
    candidate_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model_name = "RandomForestClassifier(n_estimators=200)"

model = Pipeline([
    ("prep", make_preprocessor()),
    ("clf", candidate_clf),
])

baseline.fit(X_train, y_train)
model.fit(X_train, y_train)

baseline_f1 = f1_score(y_val, baseline.predict(X_val), average="binary", zero_division=0)
model_f1 = f1_score(y_val, model.predict(X_val), average="binary", zero_division=0)
gate_pass = model_f1 >= baseline_f1 + MARGIN

metrics = {
    "metric": "f1",
    "baseline_name": "DummyClassifier(strategy='most_frequent')",
    "model_name": model_name,
    "baseline_score": round(float(baseline_f1), 4),
    "model_score": round(float(model_f1), 4),
    "margin": MARGIN,
    "gate_expression": "model_score >= baseline_score + margin",
    "gate_pass": bool(gate_pass),
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "features": FEATURES,
    "target": TARGET,
}

(ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

if not gate_pass:
    fail(
        f"Quality gate failed: model F1 {model_f1:.4f} "
        f"< baseline {baseline_f1:.4f} + margin {MARGIN:.2f}"
    )

bundle = {
    "pipeline": model,
    "feature_columns": FEATURES,
    "target_column": TARGET,
    "target_classes": sorted(y.unique().tolist()),
}
joblib.dump(bundle, ARTIFACTS / "model.joblib")
print(f"Saved model to {ARTIFACTS / 'model.joblib'}")