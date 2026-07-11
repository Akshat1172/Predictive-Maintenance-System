import pytest

from src.models.predict import (
    predict_failure,
    validate_machine_data,
)


def test_validate_machine_data_success(
    sample_machine_data,
) -> None:
    """
    Verify that valid machine data passes validation.
    """

    validate_machine_data(sample_machine_data)


def test_validate_machine_data_missing_feature(
    sample_machine_data,
) -> None:
    """
    Verify that missing required features raise ValueError.
    """

    invalid_machine_data = sample_machine_data.copy()

    invalid_machine_data.pop("Torque [Nm]")

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        validate_machine_data(
            invalid_machine_data,
        )


def test_validate_machine_data_invalid_type(
    sample_machine_data,
) -> None:
    """
    Verify that an invalid machine type raises ValueError.
    """

    invalid_machine_data = sample_machine_data.copy()

    invalid_machine_data["Type"] = "X"

    with pytest.raises(
        ValueError,
        match="Type must be one of",
    ):
        validate_machine_data(
            invalid_machine_data,
        )


def test_validate_machine_data_invalid_numeric(
    sample_machine_data,
) -> None:
    """
    Verify that non-numeric values raise ValueError.
    """

    invalid_machine_data = sample_machine_data.copy()

    invalid_machine_data["Torque [Nm]"] = "High"

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        validate_machine_data(
            invalid_machine_data,
        )


def test_predict_failure_success(
    sample_machine_data,
    saved_model_path,
) -> None:
    """
    Verify that prediction succeeds
    and returns an integer label.
    """

    prediction = predict_failure(
        sample_machine_data,
        saved_model_path,
    )

    assert isinstance(
        prediction,
        int,
    )

    assert prediction in {0, 1}


def test_predict_failure_missing_feature(
    sample_machine_data,
    saved_model_path,
) -> None:
    """
    Verify that prediction fails when
    a required feature is missing.
    """

    invalid_machine_data = sample_machine_data.copy()

    invalid_machine_data.pop("Torque [Nm]")

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        predict_failure(
            invalid_machine_data,
            saved_model_path,
        )
        
def test_predict_failure_invalid_type(
    sample_machine_data,
    saved_model_path,
) -> None:
    invalid_machine_data = sample_machine_data.copy()

    invalid_machine_data["Type"] = "X"

    with pytest.raises(
        ValueError,
        match="Type must be one of",
    ):
        predict_failure(
            invalid_machine_data,
            saved_model_path,
        )
        
def test_predict_failure_invalid_numeric(
    sample_machine_data,
    saved_model_path,
) -> None:
    invalid_machine_data = sample_machine_data.copy()

    invalid_machine_data["Torque [Nm]"] = "High"

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        predict_failure(
            invalid_machine_data,
            saved_model_path,
        )