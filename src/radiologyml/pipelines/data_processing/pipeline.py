from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_liver_dcms, create_test_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   
            node(
                func=create_test_data,
                inputs=["params:dcm_path","liver_target_df","params:nr_test_patients"],
                outputs="liver_test_df",
                name="create_test_data_node",
            ),
            node(
                func=preprocess_liver_dcms,
                inputs=["params:dcm_path","liver_target_df","liver_test_df",
                        "params:outdir","params:size","params:ssl","params:cores",
                        "params:crop","params:file_ending"],
                outputs=None,
                name="preprocess_liver_dcms_node",
            ),
        ]
    )
