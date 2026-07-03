"""
Preprocessing pipeline for the Predictive Maintenance project.

This module creates and fits a preprocessing pipeline using
Scikit-learn's ColumnTransformer.

Features:
- One-hot encoding for categorical features.
- Standard scaling for numerical features.
- Suitable for use in production ML pipelines.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------------------

CATEGORICAL_FEATURES: list[str] = [
    "Type",
]

NUMERICAL_FEATURES: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def create_preprocessor() -> ColumnTransformer:
    """
    Create and fit the preprocessing pipeline.

    The preprocessing pipeline performs:
        - One-hot encoding on categorical features.
        - Standard scaling on numerical features.

    Returns
    -------
    ColumnTransformer
        An unfitted preprocessing pipeline. Fit this pipeline only on
        the training data to avoid data leakage.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore",
                    sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor