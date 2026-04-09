"""
Flask Web Application for Building Energy Efficiency Prediction
Run: python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from pipeline.data_loader import DataLoader
from pipeline.preprocessor import Preprocessor
from pipeline.model_trainer import ModelTrainer, MODEL_REGISTRY
from pipeline.evaluator import Evaluator
from utils.logger import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global state
TRAINED_MODELS = {}
DATASETS = {}
RESULTS = {}
IS_TRAINED = False


def train_all_models(data_path="data/energy_efficiency_sample.csv"):
    """Train models on startup or on demand."""
    global TRAINED_MODELS, DATASETS, RESULTS, IS_TRAINED

    logger.info("Training models...")
    loader = DataLoader(data_path)
    df = loader.load()

    preprocessor = Preprocessor(test_size=0.2, random_state=42)
    DATASETS = preprocessor.fit_transform(df, target="both")

    # Train a fast subset of models for the web app (no grid search for speed)
    fast_models = ["linear", "ridge", "decision_tree", "random_forest", "knn"]
    trainer = ModelTrainer(model_selection="all", random_state=42, tune=False)
    trainer.model_keys = fast_models
    TRAINED_MODELS = trainer.train(DATASETS)

    evaluator = Evaluator(output_dir="results/")
    RESULTS = evaluator.evaluate(TRAINED_MODELS, DATASETS)

    IS_TRAINED = True
    logger.info("Models ready!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict heating and cooling loads from building features."""
    if not IS_TRAINED:
        return jsonify({"error": "Models not trained yet. Hit /api/train first."}), 503

    data = request.get_json()
    feature_names = DATASETS["feature_names"]
    scaler = DATASETS["scaler_X"]

    # Build feature vector in correct order
    X = np.array([[float(data.get(f, 0)) for f in feature_names]])
    X_scaled = scaler.transform(X)

    predictions = {}
    for target in ["heating", "cooling"]:
        if target in TRAINED_MODELS:
            predictions[target] = {}
            for model_key, model in TRAINED_MODELS[target].items():
                pred = model.predict(X_scaled)[0]
                predictions[target][model_key] = round(float(pred), 2)

    return jsonify(predictions)


@app.route("/api/results")
def get_results():
    """Return evaluation results."""
    if not RESULTS:
        return jsonify({"error": "No results yet."}), 503

    serializable = {}
    for target, models in RESULTS.items():
        serializable[target] = {}
        for model_key, metrics in models.items():
            serializable[target][model_key] = {k: round(float(v), 4) for k, v in metrics.items()}
    return jsonify(serializable)


@app.route("/api/train", methods=["POST"])
def train():
    """Trigger model training."""
    data_path = request.get_json().get("data_path", "data/energy_efficiency_sample.csv")
    try:
        train_all_models(data_path)
        return jsonify({"status": "ok", "models": list(TRAINED_MODELS.get("heating", {}).keys())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Auto-train on startup if data exists
    default_data = "data/energy_efficiency_sample.csv"
    if os.path.exists(default_data):
        train_all_models(default_data)
    else:
        logger.warning(f"No data at {default_data}. Start the app and POST to /api/train.")

    app.run(debug=True, host="0.0.0.0", port=5000)
