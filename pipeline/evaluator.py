"""
Model Evaluation Module
Implements all evaluation metrics mentioned in the proposal:
  CV, MBE, MSE, RMSE, MSPE, MAPE, MAE, R²
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def mean_bias_error(y_true, y_pred):
    """MBE: determines over/under estimation."""
    return np.mean(y_pred - y_true)


def coefficient_of_variance(y_true, y_pred):
    """CV of RMSE: variation of estimation error relative to true mean."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return (rmse / np.mean(y_true)) * 100


def mean_squared_percentage_error(y_true, y_pred):
    """MSPE: percentage MSE."""
    return np.mean(((y_true - y_pred) / y_true) ** 2) * 100


class Evaluator:
    """Evaluates trained models and generates comparison reports."""

    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, trained_models: dict, datasets: dict) -> dict:
        """
        Evaluate all models on test data.
        Returns: {target: {model_key: {metric: value}}}
        """
        all_results = {}

        for target_name, models in trained_models.items():
            data = datasets[target_name]
            X_test = data["X_test"]
            y_test = data["y_test"]

            all_results[target_name] = {}

            for model_key, model in models.items():
                y_pred = model.predict(X_test)
                metrics = self._compute_metrics(y_test, y_pred)
                all_results[target_name][model_key] = metrics

        return all_results

    def _compute_metrics(self, y_true, y_pred) -> dict:
        """Compute all evaluation metrics."""
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MBE": mean_bias_error(y_true, y_pred),
            "CV": coefficient_of_variance(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "MSPE": mean_squared_percentage_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    def print_summary(self, results: dict):
        """Print a formatted comparison table."""
        for target, models in results.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"  RESULTS FOR: {target.upper()} LOAD")
            logger.info(f"{'='*80}")

            rows = []
            for model_key, metrics in models.items():
                rows.append({
                    "Model": model_key,
                    **{k: round(v, 4) for k, v in metrics.items()}
                })

            df = pd.DataFrame(rows).sort_values("R2", ascending=False)
            logger.info(f"\n{df.to_string(index=False)}")

            best = df.iloc[0]
            logger.info(f"\n  Best model: {best['Model']} (R² = {best['R2']:.4f})")

    def save_results(self, results: dict):
        """Save results to JSON."""
        # Convert numpy types to Python native for JSON serialization
        serializable = {}
        for target, models in results.items():
            serializable[target] = {}
            for model_key, metrics in models.items():
                serializable[target][model_key] = {
                    k: float(v) for k, v in metrics.items()
                }

        path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {path}")

    def plot_results(self, results: dict):
        """Generate comparison plots (saved as PNG)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for target, models in results.items():
                model_names = list(models.keys())
                r2_scores = [models[m]["R2"] for m in model_names]
                mae_scores = [models[m]["MAE"] for m in model_names]
                rmse_scores = [models[m]["RMSE"] for m in model_names]

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f"Model Comparison - {target.upper()} Load", fontsize=14)

                # R² scores
                colors = ["#2ecc71" if s > 0.9 else "#f39c12" if s > 0.7 else "#e74c3c"
                          for s in r2_scores]
                axes[0].barh(model_names, r2_scores, color=colors)
                axes[0].set_xlabel("R² Score")
                axes[0].set_title("R² Score (higher = better)")
                axes[0].set_xlim(0, 1.05)

                # MAE
                axes[1].barh(model_names, mae_scores, color="#3498db")
                axes[1].set_xlabel("MAE")
                axes[1].set_title("Mean Absolute Error (lower = better)")

                # RMSE
                axes[2].barh(model_names, rmse_scores, color="#9b59b6")
                axes[2].set_xlabel("RMSE")
                axes[2].set_title("Root Mean Squared Error (lower = better)")

                plt.tight_layout()
                path = os.path.join(self.output_dir, f"comparison_{target}.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Plot saved: {path}")

        except ImportError:
            logger.warning("matplotlib not installed - skipping plots.")
