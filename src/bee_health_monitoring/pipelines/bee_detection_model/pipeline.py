from kedro.pipeline import Pipeline, node

from .nodes import load_bee_detection_model, run_detector, filter_bees

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=run_detector,
                inputs='sample_hive_image',
                outputs='result',
                name='run_obj_detection_model',
            ),
            node(
                func=filter_bees,
                inputs='result',
                outputs='result_bees',
                name='filter_bees'
            ),
        ]
    )