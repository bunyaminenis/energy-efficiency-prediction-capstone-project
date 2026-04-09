"""
Data Loading Module
Handles loading building energy dataset from CSV/Excel files.

Expected columns (based on UCI Energy Efficiency Dataset):
  Input:
    - X1: Relative Compactness
    - X2: Surface Area
    - X3: Wall Area
    - X4: Roof Area
    - X5: Overall Height
    - X6: Orientation
    - X7: Glazing Area
    - X8: Glazing Area Distribution
  Output:
    - Y1: Heating Load
    - Y2: Cooling Load
"""

import os
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Standard column mapping
COLUMN_NAMES = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}

INPUT_FEATURES = [
    "relative_compactness", "surface_area", "wall_area", "roof_area",
    "overall_height", "orientation", "glazing_area", "glazing_area_distribution"
]
TARGET_HEATING = "heating_load"
TARGET_COOLING = "cooling_load"


class DataLoader:
    """Loads and performs initial validation on the building energy dataset."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load(self) -> pd.DataFrame:
        ext = os.path.splitext(self.filepath)[1].lower()

        if ext == ".csv":
            self.df = pd.read_csv(self.filepath)
        elif ext in (".xls", ".xlsx"):
            self.df = pd.read_excel(self.filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .csv or .xlsx")

        # Rename columns if they match X1..X8, Y1, Y2 pattern
        if "X1" in self.df.columns:
            self.df.rename(columns=COLUMN_NAMES, inplace=True)

        # Also handle lowercase / alternative naming
        col_lower = {c: c.lower().replace(" ", "_") for c in self.df.columns}
        self.df.rename(columns=col_lower, inplace=True)

        self._validate()
        logger.info(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def _validate(self):
        """Basic validation checks."""
        if self.df is None or self.df.empty:
            raise ValueError("Dataset is empty.")

        missing = self.df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values detected:\n{missing[missing > 0]}")

        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows.")

    def summary(self):
        """Print dataset summary statistics."""
        if self.df is None:
            return
        logger.info(f"\n--- Dataset Summary ---\n"
                     f"Shape: {self.df.shape}\n"
                     f"Columns: {list(self.df.columns)}\n"
                     f"Dtypes:\n{self.df.dtypes}\n"
                     f"Null counts:\n{self.df.isnull().sum()}\n"
                     f"Statistics:\n{self.df.describe()}")
