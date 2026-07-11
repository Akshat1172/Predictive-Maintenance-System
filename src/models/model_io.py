from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(model: Pipeline, model_path: Path) -> None:
    """
    Save a fitted machine learning pipeline to disk.

    Parameters
    ----------
    model : Pipeline
        The fitted Scikit-learn pipeline to save.

    model_path : Path
        Destination path for the saved model artifact.

    Returns
    -------
    None
        The function saves the model to disk and does not return a value.
    """

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )
    
def load_model(model_path: Path) -> Pipeline:
    """
    Load a fitted machine learning pipeline from disk.

    Parameters
    ----------
    model_path : Path
        Path to the saved model artifact.

    Returns
    -------
    Pipeline
        The loaded fitted Scikit-learn pipeline.
    """

    model = joblib.load(model_path)

    return model