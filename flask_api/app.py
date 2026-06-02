import io
import logging
import os
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.pkl"
SENTIMENT_LABELS = {
    -1: "Negative",
    0: "Neutral",
    1: "Positive",
}


app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flask_api")


def preprocess_comment(comment: str) -> str:
    comment = str(comment).lower().strip()
    comment = re.sub(r"\n", " ", comment)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

    stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
    comment = " ".join(word for word in comment.split() if word not in stop_words)

    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word) for word in comment.split())


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)
    logger.info("Loaded vectorizer from %s", VECTORIZER_PATH)
    return model, vectorizer


def normalize_comments_payload(data: dict) -> list[str]:
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")

    comments = data.get("comments")
    if not comments:
        raise ValueError("No comments provided")
    if not isinstance(comments, list):
        raise ValueError("'comments' must be a list of strings")
    if not all(isinstance(comment, str) and comment.strip() for comment in comments):
        raise ValueError("Each comment must be a non-empty string")

    return comments


def normalize_timestamp_payload(data: dict) -> tuple[list[str], list[str]]:
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")

    comments_data = data.get("comments")
    if not comments_data:
        raise ValueError("No comments provided")
    if not isinstance(comments_data, list):
        raise ValueError("'comments' must be a list of objects")

    comments = []
    timestamps = []
    for item in comments_data:
        if not isinstance(item, dict):
            raise ValueError("Each comment item must be an object")
        comment = item.get("text")
        timestamp = item.get("timestamp")
        if not isinstance(comment, str) or not comment.strip():
            raise ValueError("Each comment item must include non-empty 'text'")
        if timestamp is None:
            raise ValueError("Each comment item must include 'timestamp'")
        comments.append(comment)
        timestamps.append(str(timestamp))

    return comments, timestamps


def predict_sentiments(comments: list[str]) -> list[int]:
    preprocessed_comments = [preprocess_comment(comment) for comment in comments]
    features = vectorizer.transform(preprocessed_comments)
    return model.predict(features).tolist()


model, vectorizer = load_artifacts()


@app.route("/")
def home():
    return jsonify({"message": "YouTube sentiment API is running"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_path": str(MODEL_PATH),
            "vectorizer_path": str(VECTORIZER_PATH),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        comments = normalize_comments_payload(request.get_json(silent=True))
        predictions = predict_sentiments(comments)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {error}"}), 500

    response = [
        {
            "comment": comment,
            "sentiment": int(sentiment),
            "sentiment_label": SENTIMENT_LABELS.get(int(sentiment), "Unknown"),
        }
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    try:
        comments, timestamps = normalize_timestamp_payload(request.get_json(silent=True))
        predictions = predict_sentiments(comments)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        logger.exception("Prediction with timestamps failed")
        return jsonify({"error": f"Prediction failed: {error}"}), 500

    response = [
        {
            "comment": comment,
            "sentiment": int(sentiment),
            "sentiment_label": SENTIMENT_LABELS.get(int(sentiment), "Unknown"),
            "timestamp": timestamp,
        }
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json(silent=True) or {}
        sentiment_counts = data.get("sentiment_counts")
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            return jsonify({"error": "Sentiment counts sum to zero"}), 400

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=["#36A2EB", "#C9CBCF", "#FF6384"],
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype="image/png")
    except Exception as error:
        logger.exception("Chart generation failed")
        return jsonify({"error": f"Chart generation failed: {error}"}), 500


@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        comments = normalize_comments_payload(request.get_json(silent=True))
        text = " ".join(preprocess_comment(comment) for comment in comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        logger.exception("Word cloud generation failed")
        return jsonify({"error": f"Word cloud generation failed: {error}"}), 500


@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json(silent=True) or {}
        sentiment_data = data.get("sentiment_data")
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        required_columns = {"timestamp", "sentiment"}
        if not required_columns.issubset(df.columns):
            return jsonify({"error": "sentiment_data must include timestamp and sentiment"}), 400

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
        df = df.dropna(subset=["timestamp", "sentiment"])
        if df.empty:
            return jsonify({"error": "No valid sentiment data provided"}), 400

        df["sentiment"] = df["sentiment"].astype(int)
        df = df.set_index("timestamp")

        monthly_counts = df.resample("ME")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: "red", 0: "gray", 1: "green"}
        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker="o",
                linestyle="-",
                label=SENTIMENT_LABELS[sentiment_value],
                color=colors[sentiment_value],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype="image/png")
    except Exception as error:
        logger.exception("Trend graph generation failed")
        return jsonify({"error": f"Trend graph generation failed: {error}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("FLASK_RUN_PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
