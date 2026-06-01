import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_params() -> dict:
    with (ROOT_DIR / "params.yaml").open("r") as file:
        return yaml.safe_load(file)["model_building"]


def load_train_data() -> pd.DataFrame:
    train_path = ROOT_DIR / "data" / "interim" / "train_processed.csv"
    df = pd.read_csv(train_path).dropna(subset=["clean_comment", "category"])
    logger.debug("Loaded training data from %s with shape %s", train_path, df.shape)
    return df


def train_model(df: pd.DataFrame, params: dict) -> tuple[LGBMClassifier, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(params["ngram_range"]),
        max_features=params["max_features"],
    )
    x_train = vectorizer.fit_transform(df["clean_comment"])
    y_train = df["category"]

    model = LGBMClassifier(
        objective="multiclass",
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        random_state=42,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    logger.debug("Model training completed")
    return model, vectorizer


def save_artifacts(model: LGBMClassifier, vectorizer: TfidfVectorizer) -> None:
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "lgbm_model.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")
    logger.debug("Saved model artifacts to %s", models_dir)


def main() -> None:
    params = load_params()
    train_df = load_train_data()
    model, vectorizer = train_model(train_df, params)
    save_artifacts(model, vectorizer)


if __name__ == "__main__":
    main()
