from pathlib import Path
from typing import Any

import pandas as pd

from src.models.save_model import load_model


REQUIRED_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def predict_failure(
    machine_data: dict[str, Any],
    model_path: Path,
) -> int:
    missing_features = (
        set(REQUIRED_FEATURES) - machine_data.keys()
    )

    if missing_features:
        raise ValueError(
            f"Missing required features: {sorted(missing_features)}"
        )

    input_df = pd.DataFrame(
        [
            {
                feature: machine_data[feature]
                for feature in REQUIRED_FEATURES
            }
        ]
    )

    model = load_model(model_path)

    prediction = model.predict(input_df)

    return int(prediction[0])