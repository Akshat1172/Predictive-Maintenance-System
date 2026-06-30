"""Utilities for loading raw project datasets."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


def load_data(
    path: Path = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the AI4I 2020 predictive maintenance dataset.

    Reads the raw CSV dataset from ``data/raw/ai4i2020.csv`` and returns both
    the original DataFrame and a deep copy for downstream transformation.

    Returns:
        A tuple containing:
            df_raw: The original dataset loaded from disk.
            df: A deep copy of ``df_raw`` for feature engineering and modeling.

    Raises:
        FileNotFoundError: If the expected raw dataset file is missing.
    """
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(
            "Dataset not found at "
            f"{RAW_DATA_PATH}. Place ai4i2020.csv in data/raw/ before loading."
        )

    df_raw = pd.read_csv(RAW_DATA_PATH)
    df = df_raw.copy(deep=True)

    return df_raw, df
