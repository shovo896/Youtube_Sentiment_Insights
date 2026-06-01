import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_artifacts():
    model = joblib.load(ROOT_DIR / "models" / "lgbm_model.pkl")
    vectorizer = joblib.load(ROOT_DIR / "models" / "tfidf_vectorizer.pkl")
    return model, vectorizer


def load_test_data() -> pd.DataFrame:
    test_path = ROOT_DIR / "data" / "interim" / "test_processed.csv"
    df = pd.read_csv(test_path).dropna(subset=["clean_comment", "category"])
    logger.debug("Loaded test data from %s with shape %s", test_path, df.shape)
    return df


def evaluate_model() -> dict:
    model, vectorizer = load_artifacts()
    test_df = load_test_data()

    x_test = vectorizer.transform(test_df["clean_comment"])
    y_test = test_df["category"]
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    logger.debug("Model evaluation completed with accuracy %.4f", metrics["accuracy"])
    return metrics


def save_metrics(metrics: dict) -> None:
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "experiment_info.json"
    with output_path.open("w") as file:
        json.dump(metrics, file, indent=2)
    logger.debug("Saved evaluation metrics to %s", output_path)


def main() -> None:
    metrics = evaluate_model()
    save_metrics(metrics)


if __name__ == "__main__":
    main()
