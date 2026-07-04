"""
Training pipeline for the Predictive Maintenance project.

This module builds reusable Scikit-learn pipelines for model training.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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
            ("preprocessor", create_preprocessor(scale_numeric=True)),
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


def build_decision_tree_pipeline() -> Pipeline:
    """
    Build an unfitted Decision Tree machine learning pipeline.

    The pipeline consists of:
        1. Data preprocessing.
        2. A Decision Tree classifier.

    Numerical features are not scaled because Decision Trees are
    invariant to feature scaling.

    Returns
    -------
    Pipeline
        An unfitted Scikit-learn Pipeline containing the
        preprocessing step and a DecisionTreeClassifier.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(scale_numeric=False),
            ),
            (
                "model",
                DecisionTreeClassifier(random_state=RANDOM_STATE,
                                       max_depth=5),
                
            ),
        ]
    )

    return pipeline

def build_random_forest_pipeline() -> Pipeline:
    """
    Build an unfitted Random Forest machine learning pipeline.

    The pipeline consists of:
        1. Data preprocessing.
        2. A Random Forest classifier.

    Numerical features are not scaled because tree-based models
    are invariant to feature scaling.

    Returns
    -------
    Pipeline
        An unfitted Scikit-learn Pipeline containing the
        preprocessing step and a RandomForestClassifier.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(scale_numeric=False),
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=100,
                    # class_weight= "balanced"
                ),
            ),
        ]
    )

    return pipeline

def build_xgboost_pipeline() -> Pipeline:
    """
    Build an unfitted XGBoost machine learning pipeline.

    The pipeline consists of:
        1. Data preprocessing.
        2. An XGBoost classifier.

    Numerical features are not scaled because XGBoost is a
    tree-based model and does not require feature scaling.

    Returns
    -------
    Pipeline
        An unfitted Scikit-learn Pipeline containing the
        preprocessing step and an XGBClassifier.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(scale_numeric=False),
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    return pipeline


