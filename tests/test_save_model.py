from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.models.model_io import save_model


def test_save_model(
    dummy_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    """
    Verify that save_model() creates the model file
    and saves a valid sklearn Pipeline.
    """

    model_path = tmp_path / "models" / "test_model.joblib"

    save_model(
        dummy_pipeline,
        model_path,
    )

    assert model_path.exists()

    loaded_pipeline = joblib.load(model_path)

    assert isinstance(
        loaded_pipeline,
        Pipeline,
    )

    assert len(loaded_pipeline.steps) == 2

    assert loaded_pipeline.steps[0][0] == "scaler"

    assert loaded_pipeline.steps[1][0] == "model"