from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.models.model_io import load_model


FEATURE_COLUMNS: Final = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

VALID_MACHINE_TYPES: Final = {
    "L",
    "M",
    "H",
}

NUMERICAL_FEATURES: Final = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def validate_machine_data(
    machine_data: dict[str, Any],
) -> None:
    """
    Validate the input machine data before prediction.

    Parameters
    ----------
    machine_data : dict[str, Any]
        Dictionary containing the machine features.

    Raises
    ------
    ValueError
        If required features are missing, the machine type is invalid,
        or a numerical feature is not numeric.
    """

    missing_features = set(FEATURE_COLUMNS).difference(
        machine_data.keys()
    )

    if missing_features:
        raise ValueError(
            f"Missing required features: {sorted(missing_features)}"
        )

    if machine_data["Type"] not in VALID_MACHINE_TYPES:
        raise ValueError(
            f"Type must be one of: {sorted(VALID_MACHINE_TYPES)}"
        )

    for feature in NUMERICAL_FEATURES:
        value = machine_data[feature]

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{feature} must be numeric."
            )


def predict_failure(
    machine_data: dict[str, Any],
    model_path: Path,
) -> int:
    """
    Predict whether a machine will fail.

    Parameters
    ----------
    machine_data : dict[str, Any]
        Dictionary containing machine feature values.

    model_path : Path
        Path to the saved model pipeline.

    Returns
    -------
    int
        0 if no failure is predicted.
        1 if failure is predicted.
    """

    validate_machine_data(machine_data)

    input_df = pd.DataFrame(
        [
            {
                feature: machine_data[feature]
                for feature in FEATURE_COLUMNS
            }
        ]
    )

    model = load_model(model_path)

    prediction = model.predict(input_df)

    return int(prediction[0])