"""
Data Preprocessing Module
Handles cleaning, normalization, feature selection, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from utils.logger import setup_logger
from pipeline.data_loader import INPUT_FEATURES, TARGET_HEATING, TARGET_COOLING

logger = setup_logger(__name__)


class Preprocessor:
    """Cleans, scales, and splits the dataset for ML training."""

    def __init__(self, test_size=0.2, random_state=42, scaler_type="standard"):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None

    def fit_transform(self, df: pd.DataFrame, target="both") -> dict:
        """
        Full preprocessing pipeline.
        Returns dict with train/test splits for each target.
        """
        df = self._clean(df)
        self.feature_names = [c for c in INPUT_FEATURES if c in df.columns]

        if not self.feature_names:
            # Fallback: use all columns except known targets
            targets_present = [c for c in [TARGET_HEATING, TARGET_COOLING] if c in df.columns]
            self.feature_names = [c for c in df.columns if c not in targets_present]

        X = df[self.feature_names].values

        # Scale features
        self.scaler_X = StandardScaler() if self.scaler_type == "standard" else MinMaxScaler()
        X_scaled = self.scaler_X.fit_transform(X)

        # Feature importance via mutual information
        datasets = {}

        targets = []
        if target in ("heating", "both") and TARGET_HEATING in df.columns:
            targets.append(("heating", TARGET_HEATING))
        if target in ("cooling", "both") and TARGET_COOLING in df.columns:
            targets.append(("cooling", TARGET_COOLING))

        if not targets:
            raise ValueError("No valid target columns found in dataset.")

        for name, col in targets:
            y = df[col].values

            # Log mutual information scores
            mi_scores = mutual_info_regression(X_scaled, y, random_state=self.random_state)
            mi_df = pd.DataFrame({
                "feature": self.feature_names,
                "mi_score": mi_scores
            }).sort_values("mi_score", ascending=False)
            logger.info(f"\nFeature importance for {name} (mutual info):\n{mi_df.to_string(index=False)}")

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y,
                test_size=self.test_size,
                random_state=self.random_state
            )

            datasets[name] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "feature_names": self.feature_names,
            }

            logger.info(f"[{name}] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        datasets["scaler_X"] = self.scaler_X
        datasets["feature_names"] = self.feature_names
        return datasets

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values, duplicates, and outliers."""
        initial_rows = len(df)

        # Drop duplicates
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows.")

        # Fill numeric NaNs with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled NaN in '{col}' with median={median_val:.4f}")

        # Drop rows that still have NaN
        df = df.dropna()

        logger.info(f"Clean dataset: {len(df)} rows")
        return df

    def inverse_transform_X(self, X_scaled):
        """Convert scaled features back to original scale."""
        if self.scaler_X:
            return self.scaler_X.inverse_transform(X_scaled)
        return X_scaled
