import os
import numpy as np
import pandas as pd
from typing import Dict
import h5py
from .liver_training import train, train_on_feats, gen_feats


def load_data(jpg_path: str, target_df: pd.DataFrame, ext: str):
    tr_files = [f"{jpg_path}-train/{f}" for f in os.listdir(f"{jpg_path}-train") if f.endswith(ext)]
    test_files = [f"{jpg_path}-test/{f}" for f in os.listdir(f"{jpg_path}-test") if f.endswith(ext)]
    test_ids = [int(f.split("/")[-1].split(f".{ext}")[0]) for f in test_files]
    
    tr_files = np.random.permutation(tr_files).tolist()
    
    target_dict = {k:v for k,v in zip(target_df.eid.values.astype(int).tolist(),target_df.mace_ever.values.astype(np.int64).tolist())}
    
    tr_targets = np.array([target_dict[int(f.split("/")[-1].split(f".{ext}")[0])] for f in tr_files ],dtype=np.int64) # if int(f.split("/")[-1].split(f".{ext}")[0]) in list(target_dict.keys())
    test_targets = np.array([target_dict[id] for id in test_ids ],dtype=np.int64) # if id in list(target_dict.keys())
    
    assert len(tr_targets)==len(tr_files), f"{len(tr_targets)=}!={len(tr_files)=}!"
    assert len(test_targets)==len(test_files), f"{len(test_targets)=}!={len(test_files)=}!"
    
    print(f"Found {len(tr_files)} patients in the train dataset!")
    print(f"Found {len(test_files)} patients in the test dataset!")
    
    for i in range(2):
        print(f"Found {len(tr_targets[tr_targets==i])} Training patients with label {i}")
        print(f"Found {len(test_targets[test_targets==i])} Test patients with label {i}")
    
    data={"tr_files":tr_files,"tr_targets":tr_targets,"test_files":test_files,"test_targets":test_targets,"test_ids":test_ids}
    
    return data

def train_liver_mace(params: Dict, data: Dict):
    """Supervised Training for liver data on MACE"""
    
    train(params,data)
    
def load_feats(params: Dict, target_df: pd.DataFrame):
    tr_files = [f"{params['feat_path']}/train/{f}" for f in os.listdir(f"{params['feat_path']}/train")if f.endswith(".h5")]
    test_files = [f"{params['feat_path']}/test/{f}" for f in os.listdir(f"{params['feat_path']}/test")if f.endswith(".h5")]
    
    test_ids = [int(f.split("/")[-1].split(".h5")[0]) for f in test_files]
    
    #tr_files = np.random.permutation(tr_files).tolist()
    
    target_dict = {k:v for k,v in zip(target_df.eid.values.astype(int).tolist(),target_df.mace_ever.values.astype(int).tolist())}
    
    tr_targets = np.array([target_dict[int(f.split("/")[-1].split(".h5")[0])] for f in tr_files],dtype=np.int64)
    test_targets = np.array([target_dict[id] for id in test_ids],dtype=np.int64)
    
    assert len(tr_targets)==len(tr_files), f"{len(tr_targets)=}!={len(tr_files)=}!"
    assert len(test_targets)==len(test_files), f"{len(test_targets)=}!={len(test_files)=}!"
    
    print(f"Found {len(tr_files)} patients in the train dataset!")
    print(f"Found {len(test_files)} patients in the test dataset!")
    
    for i in range(2):
        print(f"Found {len(tr_targets[tr_targets==i])} Training patients with label {i}")
        print(f"Found {len(test_targets[test_targets==i])} Test patients with label {i}")
    
    data={"tr_files":tr_files,"tr_targets":tr_targets,"test_files":test_files,"test_targets":test_targets,"test_ids":test_ids,"target_dict":target_dict}
    
    return data

def train_liver_mace_on_feats(params: Dict, data: Dict, target_df: pd.DataFrame):
    
    if params["gen_feats"]:
        gen_feats(params, data)
    
    data = load_feats(params, target_df)
    
    train_on_feats(params,data)
