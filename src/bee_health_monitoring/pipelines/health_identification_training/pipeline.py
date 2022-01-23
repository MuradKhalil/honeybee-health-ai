from kedro.pipeline import Pipeline, node
from .nodes import generate_X, generate_y, split_data, train_model, evaluate_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=generate_X,
                inputs=['single_bees'],
                outputs=['ids', 'tensors'],
                name='generate_X_node',
            ),
            node(
                func=generate_y,
                inputs=['single_bees_metadata', 'ids'],
                outputs=['y'],
                name="generate_y_node",
            ),
            node(
                func=split_data,
                inputs=['tensors', 'y'],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=['X_train', 'y_train'],
                outputs='bee_health_model',
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=['bee_health_model', 'X_test', 'y_test'],
                outputs=None,
                name='evaluate_model_node'
            ),
        ]
    )