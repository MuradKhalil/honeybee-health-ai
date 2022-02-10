"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

from bee_health_monitoring.pipelines import bee_detection_model as bdm
from bee_health_monitoring.pipelines import health_identification_training as hit


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    bee_detection_model_prediction_pipeline = bdm.create_pipeline()

    health_identification_training_pipeline = hit.create_pipeline()

    return {
        "__default__": health_identification_training_pipeline,
        "hit": health_identification_training_pipeline,
        "bdm": bee_detection_model_prediction_pipeline
    }
