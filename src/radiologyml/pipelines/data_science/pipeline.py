from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_data,train_liver_mace,train_liver_mace_on_feats


def create_pipeline(feats_only=False) -> Pipeline:
    
    if feats_only:
        return pipeline(
            [
                node(
                    func=load_data,
                    inputs=["params:pic_path", "liver_target_df","params:ext"],
                    outputs="data",
                    name="load_data_node",
                ),
                node(
                func=train_liver_mace_on_feats,
                inputs=["params:hps","data","liver_target_df"], 
                outputs=None,
                name="train_liver_node"
                ),
            ]
        )
    
    else:
        return pipeline(
            [
                node(
                    func=load_data,
                    inputs=["params:pic_path", "liver_target_df", "params:ext"],
                    outputs="data",
                    name="load_data_node",
                ),
                node(
                func=train_liver_mace,
                inputs=["params:hps","data"], 
                outputs=None,
                name="train_liver_node"
                ),
            ]
        )
