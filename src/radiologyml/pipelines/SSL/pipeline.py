from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_ssl

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   
            node(
                func=run_ssl,
                inputs="params:args",
                outputs=None,
                name="run_ssl_node",
            ),
        ]
    )