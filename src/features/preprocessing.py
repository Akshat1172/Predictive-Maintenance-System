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


def create_preprocessor(scale_numeric: bool = True,) -> ColumnTransformer:
    """
    Create and fit the preprocessing pipeline.

        Parameters
        ----------
        scale_numeric : bool, default=True
            If True, apply StandardScaler to numerical features.
            If False, pass numerical features through unchanged.

    Returns
    -------
    ColumnTransformer
        An unfitted preprocessing pipeline. Fit this pipeline only on
        the training data to avoid data leakage.
    """
    numerical_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

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
                numerical_transformer,
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor