from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from src.models.model_io import (
    load_model,
    save_model,
)


def test_load_model_success(
    dummy_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    """
    Verify that load_model() correctly loads
    a saved sklearn Pipeline.
    """

    model_path = tmp_path / "models" / "test_model.joblib"

    save_model(
        dummy_pipeline,
        model_path,
    )

    loaded_pipeline = load_model(
        model_path,
    )

    assert isinstance(
        loaded_pipeline,
        Pipeline,
    )
    assert len(loaded_pipeline.steps) == 2
    assert loaded_pipeline.steps[-1][0] == "model"


def test_load_model_file_not_found() -> None:
    """
    Verify that loading a missing model
    raises FileNotFoundError.
    """

    with pytest.raises(FileNotFoundError):
        load_model(
            Path("does_not_exist.joblib"),
        )