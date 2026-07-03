"""
Training pipeline for the Predictive Maintenance project.

This module builds reusable Scikit-learn pipelines for model training.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import create_preprocessor


RANDOM_STATE = 42
MAX_ITER = 1000


def build_logistic_pipeline() -> Pipeline:
    """
    Create a machine learning pipeline for Logistic Regression.

    The pipeline performs:
        1. Data preprocessing.
        2. Logistic Regression training.

    Returns
    -------
    Pipeline
        An unfitted Scikit-learn Pipeline.
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=MAX_ITER,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return pipeline
