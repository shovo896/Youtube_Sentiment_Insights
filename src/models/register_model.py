import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from sklearn.pipeline import Pipeline


logger = logging.getLogger("register_model")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = "http://13.220.34.174:5000"
DEFAULT_EXPERIMENT_NAME = "Youtube Sentiment Insights"
DEFAULT_REGISTERED_MODEL_NAME = "youtube_sentiment_lgbm"


def load_env_file() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return

    with env_path.open("r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))

    if os.getenv("aws_access_key") and not os.getenv("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["aws_access_key"]
    if os.getenv("aws_secret_access_key") and not os.getenv("AWS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["aws_secret_access_key"]


def load_json(path: Path) -> dict:
    with path.open("r") as file:
        return json.load(file)


def load_model_pipeline() -> Pipeline:
    model = joblib.load(ROOT_DIR / "models" / "lgbm_model.pkl")
    vectorizer = joblib.load(ROOT_DIR / "models" / "tfidf_vectorizer.pkl")
    return Pipeline(
        [
            ("tfidf_vectorizer", vectorizer),
            ("model", model),
        ]
    )


def load_model_params() -> dict:
    params_path = ROOT_DIR / "params.yaml"
    if not params_path.exists():
        return {}

    with params_path.open("r") as file:
        params = yaml.safe_load(file) or {}
    return params.get("model_building", {})


def log_metrics(metrics: dict, prefix: str = "") -> None:
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}".replace(" ", "_").replace("-", "minus_")
        if isinstance(value, dict):
            log_metrics(value, prefix=f"{metric_name}_")
        elif isinstance(value, (int, float)):
            mlflow.log_metric(metric_name, float(value))


def main() -> None:
    load_env_file()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    registered_model_name = os.getenv("REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("Using MLflow tracking URI: %s", mlflow.get_tracking_uri())
    logger.info("Using MLflow experiment: %s", experiment_name)

    experiment_info_path = ROOT_DIR / "reports" / "experiment_info.json"
    experiment_info = load_json(experiment_info_path)
    pipeline = load_model_pipeline()
    params = load_model_params()

    accuracy = experiment_info.get("accuracy")
    logger.debug("Loaded experiment info from %s", experiment_info_path)

    with mlflow.start_run(run_name="register_lgbm_model") as run:
        mlflow.log_params(params)
        log_metrics(experiment_info)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        logger.info("Registered model '%s' from run_id=%s", registered_model_name, run.info.run_id)
        logger.info("Model registration completed. Accuracy: %s", accuracy)


if __name__ == "__main__":
    main()
