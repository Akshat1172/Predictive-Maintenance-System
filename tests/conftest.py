from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features.preprocessing import create_preprocessor
from src.models.model_io import save_model

@pytest.fixture
def dummy_pipeline() -> Pipeline:
    """
    Create a simple sklearn pipeline for testing.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier()),
        ]
    )


@pytest.fixture
def sample_machine_data() -> dict[str, object]:
    """
    Valid machine input used in prediction tests.
    """
    return {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 120,
    }


@pytest.fixture
def saved_model_path(
    trained_dummy_pipeline: Pipeline,
    tmp_path: Path,
) -> Path:
    """
    Save a dummy pipeline to a temporary location.
    """

    model_path = tmp_path / "models" / "test_model.joblib"

    save_model(
        trained_dummy_pipeline,
        model_path,
    )

    return model_path

@pytest.fixture
def trained_dummy_pipeline() -> Pipeline:
    """
    Create a fitted pipeline for prediction tests.
    """

    X_train = pd.DataFrame(
        [
            {
                "Type": "L",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 120,
            },
            {
                "Type": "M",
                "Air temperature [K]": 300.2,
                "Process temperature [K]": 309.5,
                "Rotational speed [rpm]": 1600,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 80,
            },
        ]
    )

    y_train = [0, 1]

    pipeline = Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline