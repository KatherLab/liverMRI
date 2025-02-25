
import pydicom as dicom
import zipfile
from skimage.color import gray2rgb
from PIL import Image
import os
import multiprocessing as mp
from os.path import exists
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from functools import partial
import pandas as pd

def save_img(file: str, outdir: str, size: int, ssl:bool, crop:bool, ext:str):
    """loads zip file containing liver dcm file, converts the first MRI (with highest contrast) to jpg and saves it"""
    
    arch = zipfile.ZipFile(file, 'r')
    pics = [dicom.dcmread(arch.open(a)) for a in arch.namelist() if a.endswith("dcm")]
    pics = [p for p in pics if len(p.ImageComments.split(' '))>1]
    TIeff_values = {k:v for k,v in zip([int(p.ImageComments.split(' ')[1]) for p in pics if p.ImageType[2]=='M'],[p for p in pics if p.ImageType[2]=="M"])}
    if len(TIeff_values.keys())==0:
        print(f"Could not find M image for {file}!")
    else:
        img = []
        img = TIeff_values[sorted(list(TIeff_values.keys()))[0]].pixel_array
        if len(img)>0:
            if size!="None":
                if crop:
                    transform = transforms.CenterCrop(size)
                    image_crop = transform(Image.fromarray(np.uint8(img)))
                    img = np.array(image_crop)
                else:
                    img = np.array(Image.fromarray(np.uint8(img)).resize((size,size)))
            img = gray2rgb(img)
            if ssl:
                if not exists(f"{outdir}/{file.split('/')[-1].split('_')[0]}"):
                    os.mkdir(f"{outdir}/{file.split('/')[-1].split('_')[0]}")
                if not exists(f"{outdir}/{file.split('/')[-1].split('_')[0]}/{file.split('/')[-1].split('_')[0]}.{ext}"):
                    Image.fromarray(np.uint8(img)).save(f"{outdir}/{file.split('/')[-1].split('_')[0]}/{file.split('/')[-1].split('_')[0]}.{ext}")
                else:
                    Image.fromarray(np.uint8(img)).save(f"{outdir}/{file.split('/')[-1].split('_')[0]}/{file.split('/')[-1].split('_')[0]}-1.{ext}")
            else:
                Image.fromarray(np.uint8(img)).save(f"{outdir}/{file.split('/')[-1].split('_')[0]}.{ext}")
        else: 
            print(f"Could not find suitable MRI image for {file}")
            
def preprocess_liver_dcms(dcm_path: str, df: pd.DataFrame, test_df: pd.DataFrame, 
                          outdir:str, size:int, ssl: bool, cores: int, crop: bool, ext: str):
    """Generate list of train + test patients and save pics in suitable directories"""
    
    if ssl:
        files = [f"{dcm_path}/{f}" for f in os.listdir(dcm_path) if f.endswith(".zip")]
    else:
        files = [f"{dcm_path}/{f}" for f in os.listdir(dcm_path) if f.endswith("_2_0.zip")]
    
    test_ids = test_df.id.values.astype(int).tolist()
    
    print(f"{len(files)=}")
    
    train_files = [f for f in files if int(f.split("/")[-1].split("_")[0]) not in test_ids]
    
    print(f"{len(train_files)=}")
    
    if not ssl:
        target_ids = df.eid.values.astype(int).tolist()
        test_files = [f for f in files if int(f.split("/")[-1].split("_")[0]) in test_ids] # int(f.split("/")[-1].split("_")[0]) in target_ids and
        train_files = [f for f in train_files if int(f.split("/")[-1].split("_")[0]) in target_ids and int(f.split("/")[-1].split("_")[0]) not in test_ids]
        print(f"Nr test files: {len(test_files)}") 
    
    assert len(test_ids) == len(list(set(test_ids))), f"Veto ids are not unique!"
    
    assert len([f for f in test_files if f.split("/")[-1] in [p.split("/")[-1] for p in train_files]])==0, f"Huge problem: {[f for f in test_files if f.split('/')[-1] in [p.split('/')[-1] for p in train_files]]}"
    
    save_train_img_partial = partial(save_img, outdir=f"{outdir}-train", size=size, ssl=ssl, crop=crop, ext=ext)
    
    if not exists(f"{outdir}-train"):
        os.mkdir(f"{outdir}-train")
    
    with mp.Pool() as pool, tqdm(total=len(train_files)) as pbar:
         for _ in pool.imap_unordered(save_train_img_partial, train_files):
            pbar.update(1)
    
    if not ssl:    
        save_test_img_partial = partial(save_img, outdir=f"{outdir}-test", size=size, ssl=ssl, crop=crop, ext=ext)
        if not exists(f"{outdir}-test"):
            os.mkdir(f"{outdir}-test")
        with mp.Pool(cores) as pool, tqdm(total=len(test_files)) as pbar:
            for _ in pool.imap_unordered(save_test_img_partial, test_files):
                pbar.update(1)
                
def create_test_data(dcm_path: str, df: pd.DataFrame, total_test_patients: int):
    
    
    files = [f"{dcm_path}/{f}" for f in os.listdir(dcm_path) if f.endswith("_2_0.zip")]
    ids = df.eid.values.astype(int).tolist()
    
    available_ids = [int(f.split("/")[-1].split("_")[0]) for f in files if int(f.split("/")[-1].split("_")[0]) in ids]
    
    test_ids = list(set(df[(df["mace_or_cv_death_after_imaging"]==1) & (df["eid"].isin(available_ids))]["eid"].values.astype(int).tolist()))
    test_ids += list(set(np.random.permutation(df[(df["mace_or_cv_death_after_imaging"]==0) & (df["eid"].isin(available_ids))]["eid"].values).tolist()))[:(total_test_patients-len(test_ids))]
    assert len(list(set(test_ids)))==total_test_patients, f"{len(list(set(test_ids)))=}!={total_test_patients=}!"
    
    test_id_dict = {"id":test_ids}
    
    return pd.DataFrame(test_id_dict)
