import argparse
import json
import logging
from pathlib import Path 

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%levelname]s %(message)s")
log = logging.getLogger(__name__)

ADULT_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")

Columns = ["age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"]

Numeric_Features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

Categorical_features = ["workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"]

def load_data() -> pd.DataFrame:
    log.info("Loading Adult Dataset from UCI...")
    df = pd.read_csv(ADULT_URL, header=None, names=Columns, na_values=" ?", skipinitialspace=True)
    df = df.dropna().reset_index(drop=True)
    df["target"] = (df["income"].str.strip() == ">50k").astype(int)
    df = df.drop(columns=["income"])
    log.info(f"Loaded {len(df)} rows after dropping missing values.")
    return df

def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, Numeric_Features),
        ("cat", categorical_transformer, Categorical_features)
    ])

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

def main(output_dir: Path, seed: int) -> None:
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df[Numeric_Features + Categorical_features]
    y = df["target"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)
    log.info(f"Split sizes - train: {len(X_train)}, calib: {len(X_calib)}, test: {len(X_test)}")

    pipeline= build_pipeline()
    log.info("Fitting model...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    log.info(f"Test accuracy: {acc:.4f}")

    model_path = output_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    log.info(f"Saved model to {model_path}")

    X_calib.assign(target=y_calib.values).to_csv(output_dir / "catlib.csv", index=False)
    X_test.assign(target=y_test.values).to_csv(output_dir / "test.csv", index=False)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"accuracy": acc, "report": report, "seed": seed}, f, indent=2)
        log.info(f"Saved calibration/test splits and metrics.json to {output_dir}")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train a simple classifier on the UCI Adult dataset.")
        parser.add_argument("--output-dir", type=Path, default=Path("./artifacts"))
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()
        main(args.output_dir, args.seed)
