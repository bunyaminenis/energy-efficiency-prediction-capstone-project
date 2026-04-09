"""
Predictor Module
Handles model saving/loading and interactive predictions.
"""

import os
import joblib
import numpy as np
from utils.logger import setup_logger
from pipeline.data_loader import INPUT_FEATURES

logger = setup_logger(__name__)


class Predictor:
    """Loads saved models and makes predictions on new building data."""

    def __init__(self, model_dir="results/"):
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None

    @staticmethod
    def save_models(trained_models: dict, scaler, feature_names: list, output_dir="models/"):
        """Save all trained models and the scaler to disk."""
        os.makedirs(output_dir, exist_ok=True)

        joblib.dump(scaler, os.path.join(output_dir, "scaler_X.joblib"))
        joblib.dump(feature_names, os.path.join(output_dir, "feature_names.joblib"))

        for target, models in trained_models.items():
            for model_key, model in models.items():
                path = os.path.join(output_dir, f"{target}_{model_key}.joblib")
                joblib.dump(model, path)
                logger.info(f"Saved: {path}")

    def load_model(self, target: str, model_key: str):
        """Load a specific model from disk."""
        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler_X.joblib"))
        path = os.path.join(self.model_dir, f"{target}_{model_key}.joblib")
        model = joblib.load(path)
        self.models[f"{target}_{model_key}"] = model
        return model

    def predict(self, building_features: dict, target="heating", model_key="random_forest"):
        """
        Predict energy load for a single building.

        Args:
            building_features: dict with keys matching INPUT_FEATURES
            target: 'heating' or 'cooling'
            model_key: which model to use

        Returns:
            Predicted load value (kWh/m²)
        """
        key = f"{target}_{model_key}"
        if key not in self.models:
            self.load_model(target, model_key)

        feature_names = joblib.load(os.path.join(self.model_dir, "feature_names.joblib"))
        X = np.array([[building_features.get(f, 0) for f in feature_names]])
        X_scaled = self.scaler.transform(X)

        prediction = self.models[key].predict(X_scaled)
        return prediction[0]

    def predict_batch(self, X: np.ndarray, target="heating", model_key="random_forest"):
        """Predict for multiple buildings at once."""
        key = f"{target}_{model_key}"
        if key not in self.models:
            self.load_model(target, model_key)

        X_scaled = self.scaler.transform(X)
        return self.models[key].predict(X_scaled)

    def predict_interactive(self):
        """Interactive CLI prediction mode."""
        logger.info("\n--- Interactive Prediction Mode ---")
        logger.info("Enter building characteristics (or 'quit' to exit):\n")

        feature_names = joblib.load(os.path.join(self.model_dir, "feature_names.joblib"))

        while True:
            features = {}
            for feat in feature_names:
                val = input(f"  {feat}: ")
                if val.lower() == "quit":
                    return
                features[feat] = float(val)

            for target in ["heating", "cooling"]:
                try:
                    pred = self.predict(features, target=target, model_key="random_forest")
                    logger.info(f"  Predicted {target} load: {pred:.2f} kWh/m²")
                except Exception as e:
                    logger.error(f"  Could not predict {target}: {e}")

            cont = input("\nPredict another? (y/n): ")
            if cont.lower() != "y":
                break
