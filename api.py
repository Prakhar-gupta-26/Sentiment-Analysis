from pathlib import Path
from functools import lru_cache
from io import BytesIO
import base64
import re
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"

# NLTK stopwords must be downloaded once:
# python -c "import nltk; nltk.download('stopwords')"
STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)


@lru_cache(maxsize=1)
def load_models():
    """Load the trained model files once and reuse them."""
    with open(MODELS_DIR / "model_xgb.pkl", "rb") as f:
        predictor = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "countVectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    return predictor, scaler, cv


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        predictor, scaler, cv = load_models()

        # Bulk prediction: uploaded CSV file
        if "file" in request.files:
            uploaded_file = request.files["file"]
            if not uploaded_file.filename:
                return jsonify({"error": "Please choose a CSV file."}), 400

            data = pd.read_csv(uploaded_file)
            if "Sentence" not in data.columns:
                return jsonify({
                    "error": "CSV must contain a column named 'Sentence'."
                }), 400
            if data.empty:
                return jsonify({"error": "The uploaded CSV has no rows."}), 400

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getvalue()
            ).decode("ascii")
            response.headers["Access-Control-Expose-Headers"] = (
                "X-Graph-Exists, X-Graph-Data"
            )
            return response

        # Single text prediction
        payload = request.get_json(silent=True) or {}
        text_input = payload.get("text", "").strip()
        if not text_input:
            return jsonify({"error": "Please enter a review to predict."}), 400

        predicted_sentiment = single_prediction(
            predictor, scaler, cv, text_input
        )
        return jsonify({"prediction": predicted_sentiment})

    except FileNotFoundError as exc:
        app.logger.exception("A required model file was not found.")
        return jsonify({
            "error": (
                f"Model file not found: {exc}. Check that the Models folder "
                "contains model_xgb.pkl, scaler.pkl and countVectorizer.pkl."
            )
        }), 500
    except Exception as exc:
        app.logger.exception("Prediction failed.")
        return jsonify({"error": str(exc)}), 500


def preprocess_text(text):
    stemmer = PorterStemmer()
    words = re.sub(r"[^a-zA-Z]", " ", str(text)).lower().split()
    words = [stemmer.stem(word) for word in words if word not in STOPWORDS]
    return " ".join(words)


def single_prediction(predictor, scaler, cv, text_input):
    corpus = [preprocess_text(text_input)]
    features = cv.transform(corpus).toarray()
    features_scaled = scaler.transform(features)
    prediction = predictor.predict_proba(features_scaled).argmax(axis=1)[0]
    return "Positive" if prediction == 1 else "Negative"


def bulk_prediction(predictor, scaler, cv, data):
    corpus = [preprocess_text(sentence) for sentence in data["Sentence"]]
    features = cv.transform(corpus).toarray()
    features_scaled = scaler.transform(features)
    predictions = predictor.predict_proba(features_scaled).argmax(axis=1)
    data = data.copy()
    data["Predicted sentiment"] = [
        sentiment_mapping(value) for value in predictions
    ]

    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)
    return predictions_csv, graph


def get_distribution_graph(data):
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = data["Predicted sentiment"].value_counts()
    counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=("green", "red"),
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "black"},
        ax=ax,
        title="Sentiment Distribution",
    )
    ax.set_ylabel("")
    graph = BytesIO()
    fig.savefig(graph, format="png", bbox_inches="tight")
    plt.close(fig)
    graph.seek(0)
    return graph


def sentiment_mapping(value):
    return "Positive" if value == 1 else "Negative"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
