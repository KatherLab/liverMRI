from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
from os.path import exists
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from sklearn.utils.class_weight import compute_class_weight
from .vits import vit_conv_base
from ..SSL.ctran import convnextv2, ctranspath, pvt_v2_b2_li, maxvit_small, maxvit_tiny
from typing import Dict
import h5py

class Liverdataset(Dataset):
    def __init__(self, img_list, targets, transform=None):
        self.img_list = img_list
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_id = self.img_list[index]
        img = np.array(Image.open(img_id))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        
        return img, self.targets[index]
    
class FoundationDataset(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_id = self.img_list[index]
        img = np.array(Image.open(img_id))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        
        return img, img_id.split("/")[-1].split(".")[0]
    
class Featuredataset(Dataset):
    def __init__(self, feat_list, ids, targets):
        self.feat_list = feat_list
        self.targets = targets
        self.ids = ids
        #self.transform = transform

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, index):
        feat_id = self.feat_list[index]
        #img = np.array(Image.open(img_id))
        feats = h5py.File(feat_id,'r')["feats"][:]
        #img = np.transpose(img, (2, 0, 1))
        feat_t = torch.from_numpy(feats).float()
        #if self.transform:
        #    img = self.transform(img)
        
        return feat_t, self.ids[int(feat_id.split("/")[-1].split(".h5")[0])]
    
class Classifier(nn.Module):
    def __init__(self, base, input_dim, num_classes, freeze_base=True):
        super(Classifier, self).__init__()

        self.base = base

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(input_dim,num_classes),
        )

    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x

class LinearProber(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProber, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim,num_classes),
        )

    def forward(self, x):
        x = self.head(x)
        return x
    
def load_base(arch:str, chkpt_path: str):
    
    if arch=="vit":
        base = vit_conv_base()
        base.head = nn.Identity()
        vit = torch.load(chkpt_path, map_location=torch.device('cpu'))
        base.load_state_dict(vit,strict=True)
        return base
            
    elif arch=="pvt":
    
        base = pvt_v2_b2_li()
        #print(base)   
        base.head = nn.Identity()
        pvt = torch.load(chkpt_path, map_location=torch.device('cpu'))
        base.load_state_dict(pvt, strict=True)
        return base
    
    elif arch=="convnext":
        
        base = convnextv2()
        base.head.fc = nn.Identity()
        convnext = torch.load(chkpt_path, map_location=torch.device('cpu'))
        base.load_state_dict(convnext, strict=True)
        return base
        
    elif arch=="swin":
        base = ctranspath()
        base.head.fc=nn.Identity()
        swin = torch.load(chkpt_path, map_location=torch.device('cpu'))
        base.load_state_dict(swin, strict=True)
        return base
    
    elif arch=="maxvit_small":
        base = maxvit_small()
        #print(base)
        base.head.fc = nn.Identity()
        mvit = torch.load(chkpt_path, map_location=torch.device('cpu'))
        base.load_state_dict(mvit, strict=True)
        return base
        
    elif arch=="maxvit_tiny":
        base = maxvit_tiny()
        base.head.fc = nn.Identity()
        mvit = torch.load(chkpt_path, map_location=torch.device('cpu'))
        base.load_state_dict(mvit, strict=True)
        return base

    else:
        print(f"The model {arch} is not configured!")

    
def train(params: Dict, data: Dict):
    """Train liver data on downstream MACE target"""
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    
    train_dataset = Liverdataset(data["tr_files"], data["tr_targets"],transform=normalize)
    test_dataset = Liverdataset(data["test_files"], data["test_targets"],transform=normalize)
    
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["cores"])
    
    kf = KFold(n_splits=params["num_folds"], shuffle=True) 
    
    for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold+1}/{params['num_folds']}")
    
        #base = vit_conv_base() 
        #base.head = nn.Identity()
        
        base = load_base(params["arch"],params["model_path"])
        model = Classifier(base, base.num_features, 2)
        
        print(f"Created model: {params['arch']} from {params['model_path']}")
        
        # Create subsets for training and validation
        train_subset = Subset(train_dataset, train_index)
        val_subset = Subset(train_dataset, val_index)

        # Create data loaders for the subsets
        train_loader = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True, num_workers=params["cores"])
        val_loader = DataLoader(val_subset, batch_size=params["batch_size"], shuffle=False, num_workers=params["cores"])

        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=float(params["lr"]),)
        
        positive_weights = compute_class_weight('balanced', classes=[0,1], y=train_subset.dataset.targets)
        positive_weights = torch.tensor(positive_weights, dtype=torch.float).cuda()
        criterion = nn.CrossEntropyLoss(weight=positive_weights)

        
        best_val_loss = float('inf')  
        best_model_state_dict = None
        
        pbar = tqdm(total=params["epochs"], desc='Training Progress', unit='epoch')

        stop_count = 0
        
        for epoch in range(params["epochs"]):
            model.train()  

            total_train_loss = 0.0
            total_train_correct = 0
            

            for images, targets in train_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()

                logits = model(images)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                _, predicted_labels = torch.max(logits, dim=1)
                total_train_correct += (predicted_labels == targets).sum().item()


            train_loss_avg = total_train_loss / len(train_loader)
            train_accuracy = total_train_correct / len(train_subset)

            model.eval() 

            total_val_loss = 0.0
            total_val_correct = 0
            
            all_labels = []
            all_preds = []
            
            for images, targets in val_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()

                with torch.no_grad():
                    logits = model(images)
                    val_loss = criterion(logits, targets)
                    total_val_loss += val_loss.item()

                    _, predicted_labels = torch.max(logits, dim=1)
                    all_preds.extend(predicted_labels.cpu().numpy().tolist())
                    all_labels.extend(targets.cpu().numpy().tolist())
                    total_val_correct += (predicted_labels == targets).sum().item()


            val_loss_avg = total_val_loss / len(val_loader)
            val_accuracy = total_val_correct / len(val_subset)


            if val_loss_avg <= best_val_loss:
                best_val_loss = val_loss_avg
                best_model_state_dict = model.state_dict()
                stop_count = 0
                tqdm.write('Epoch: {}, loss: {:.4f}, Acc.: {:.3f}%, val_loss: {:.4f}, val_acc: {:.3f}%, val_bacc: {:.3f}%'.format(
                    epoch + 1, train_loss_avg, train_accuracy * 100, val_loss_avg, val_accuracy * 100, balanced_accuracy_score(all_labels,all_preds)*100))
            else:
                stop_count += 1
                if stop_count >= params["es"]:
                    print("Early stopping triggered!")
                    break 
                  
            pbar.set_postfix(
                loss=train_loss_avg,
                acc=train_accuracy,
                val_loss=val_loss_avg,
                val_acc=val_accuracy
            )
            pbar.update(1)
        # Load the best model state dict
        model.load_state_dict(best_model_state_dict)

        # Evaluate the model on the test set
        model.eval()
        total_test_correct = 0
        all_predicted_probs = []
        all_targets = []
        all_pred_labels = []

        pred_dict = {"id":data["test_ids"],"pred":[],"MACE":data["test_targets"]}
        
        test_loss = 0
        with torch.no_grad():
            for images, targets in test_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()

                with torch.no_grad():
                    logits = model(images)
                    
                    loss = criterion(logits,targets)
                    test_loss += loss.item()
                    
                
                    _, predicted_labels = torch.max(logits, dim=1)
                    total_test_correct += (predicted_labels == targets).sum().item()

                    predicted_probs = nn.functional.softmax(logits, dim=1)
                    pred_dict["pred"].extend(predicted_probs[:,1].cpu().numpy().flatten().tolist())
                    all_predicted_probs.append(predicted_probs[:, 1].cpu().numpy())  
                    all_pred_labels.extend(predicted_labels.cpu().numpy().tolist())
                    all_targets.append(targets.cpu().numpy())

        # Calculate test accuracy
        test_accuracy = total_test_correct / len(test_dataset)

        # Flatten the predicted probabilities and targets
        all_predicted_probs = np.concatenate(all_predicted_probs)
        all_targets = np.concatenate(all_targets)

        test_loss_avg = test_loss / len(test_loader)
        
        #if test_loss_avg < min_test_loss:
        min_test_loss = test_loss_avg
        torch.save(model.state_dict(),f"/path/to/data/06_models/liver_{params['arch']}_mace_{fold+1}.pth")
        
        assert len(np.unique([len(pred_dict[k]) for k in pred_dict.keys()]))==1, f"the lengths of the lists are different: {[len(pred_dict[k]) for k in pred_dict.keys()]}"
        
        test_df = pd.DataFrame(pred_dict)
        test_df.to_csv(f"/path/to/data/07_model_output/test_preds_mace_{params['arch']}_f{fold+1}.csv",index=False)
        
        # Calculate AUROC
        test_auroc = roc_auc_score(all_targets, all_predicted_probs)

        
        pbar.close()

        tqdm.write(f"Test loss: {test_loss_avg:.5f}, Test Acc: {test_accuracy*100:.2f}, AUROC: {test_auroc:.4f}, bACC: {balanced_accuracy_score(all_targets, all_pred_labels)}")

def gen_feats(params: Dict, data: Dict):
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    
    train_dataset = FoundationDataset(data["tr_files"],transform=normalize)
    test_dataset = FoundationDataset(data["test_files"],transform=normalize)
    
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size_gen"], shuffle=False, num_workers=params["cores"])
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size_gen"], shuffle=False, num_workers=params["cores"])
    
    model = load_base(params["arch"],params["model_path"])
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    model.eval()
    if not exists(f"{params['feat_path']}/train"):
        os.mkdir(f"{params['feat_path']}/train")
    if not exists(f"{params['feat_path']}/test"):
        os.mkdir(f"{params['feat_path']}/test")
    with torch.no_grad():
        for images, img_ids in tqdm(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
            logits = model(images)
            #print(list(img_ids))
            assert len(logits)==len(img_ids), f"The lenghts are not the same oO"
            for feats,id in zip(tqdm(logits.cpu().detach().numpy().tolist(),leave=False),list(img_ids)):
                file = h5py.File(f"{params['feat_path']}/train/{id}.h5", 'w')
                file.create_dataset("feats",data=feats)
                file.close()
        
        for images, img_ids in tqdm(test_loader):
            if torch.cuda.is_available():
                images = images.cuda()
            logits = model(images)
            for feats,id in zip(tqdm(logits.cpu().numpy().tolist(),leave=False),list(img_ids)):
                file = h5py.File(f"{params['feat_path']}/test/{id}.h5", 'w')
                file.create_dataset("feats",data=feats)
                file.close()
    
def train_on_feats(params: Dict,data: Dict):
    
    train_dataset = Featuredataset(data["tr_files"], data["target_dict"],data["tr_targets"])
    test_dataset = Featuredataset(data["test_files"], data["target_dict"],data["test_targets"])
    
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["cores"])
    
    
    kf = KFold(n_splits=params["num_folds"], shuffle=True) 
    
    for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold+1}/{params['num_folds']}")
    
        #base = vit_conv_base() 
        #base.head = nn.Identity()
        
        base = load_base(params["arch"],params["model_path"])
        model = LinearProber(base.num_features,2)
        del base
            
        # Create subsets for training and validation
        train_subset = Subset(train_dataset, train_index)
        val_subset = Subset(train_dataset, val_index)

        # Create data loaders for the subsets
        train_loader = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True, num_workers=params["cores"])
        val_loader = DataLoader(val_subset, batch_size=params["batch_size"], shuffle=False, num_workers=params["cores"])

        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=float(params["lr"]),)
        
        positive_weights = compute_class_weight('balanced', classes=[0,1], y=train_subset.dataset.targets)
        positive_weights = torch.tensor(positive_weights, dtype=torch.float).cuda()
        criterion = nn.CrossEntropyLoss(weight=positive_weights)

        
        best_val_loss = float('inf')  
        best_model_state_dict = None
        
        pbar = tqdm(total=params["epochs"], desc='Training Progress', unit='epoch')

        stop_count = 0
        
        for epoch in range(params["epochs"]):
            model.train()  

            total_train_loss = 0.0
            total_train_correct = 0
            

            for images, targets in train_loader:
                #print(targets)
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                _, predicted_labels = torch.max(logits, dim=1)
                total_train_correct += (predicted_labels == targets).sum().item()


            train_loss_avg = total_train_loss / len(train_loader)
            train_accuracy = total_train_correct / len(train_subset)

            model.eval() 

            total_val_loss = 0.0
            total_val_correct = 0

            all_labels = []
            all_preds = []
            
            for images, targets in val_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()

                with torch.no_grad():
                    logits = model(images)
                    val_loss = criterion(logits, targets)
                    total_val_loss += val_loss.item()

                    _, predicted_labels = torch.max(logits, dim=1)
                    total_val_correct += (predicted_labels == targets).sum().item()
                    all_preds.extend(predicted_labels.cpu().numpy().tolist())
                    all_labels.extend(targets.cpu().numpy().tolist())

            val_loss_avg = total_val_loss / len(val_loader)
            val_accuracy = total_val_correct / len(val_subset)


            if val_loss_avg <= best_val_loss:
                best_val_loss = val_loss_avg
                best_model_state_dict = model.state_dict()
                stop_count = 0
                tqdm.write('Epoch: {}, loss: {:.4f}, Acc.: {:.3f}%, val_loss: {:.4f}, val_acc: {:.3f}%, val_bacc: {:.3f}%'.format(
                    epoch + 1, train_loss_avg, train_accuracy * 100, val_loss_avg, val_accuracy * 100, balanced_accuracy_score(all_labels,all_preds)*100))
            else:
                stop_count += 1
                if stop_count >= params["es"]:
                    print("Early stopping triggered!")
                    break 
                  
            pbar.set_postfix(
                loss=train_loss_avg,
                acc=train_accuracy,
                val_loss=val_loss_avg,
                val_acc=val_accuracy
            )
            pbar.update(1)
        # Load the best model state dict
        model.load_state_dict(best_model_state_dict)

        # Evaluate the model on the test set
        model.eval()
        total_test_correct = 0
        all_predicted_probs = []
        all_targets = []
        all_pred_labels = []

        pred_dict = {"id":data["test_ids"],"pred":[],"MACE":data["test_targets"]}
        
        test_loss = 0
        
        for images, targets in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()

            with torch.no_grad():
                logits = model(images)
                
                loss = criterion(logits,targets)
                test_loss += loss.item()
                
            
                _, predicted_labels = torch.max(logits, dim=1)
                total_test_correct += (predicted_labels == targets).sum().item()

                predicted_probs = nn.functional.softmax(logits, dim=1)
                pred_dict["pred"].extend(predicted_probs[:,1].cpu().numpy().flatten().tolist())

                all_predicted_probs.append(predicted_probs[:, 1].cpu().numpy())  
                all_pred_labels.extend(predicted_labels.cpu().numpy().tolist())
                all_targets.append(targets.cpu().numpy())

        # Calculate test accuracy
        test_accuracy = total_test_correct / len(test_dataset)

        # Flatten the predicted probabilities and targets
        all_predicted_probs = np.concatenate(all_predicted_probs)
        all_targets = np.concatenate(all_targets)

        test_loss_avg = test_loss / len(test_loader)
        
        #if test_loss_avg < min_test_loss:
        min_test_loss = test_loss_avg
        torch.save(model.state_dict(),f"/mnt/bulk/timlenz/tumpe/RadiologyMLStation/radiologyml/data/06_models/liver_{params['arch']}_mace_{fold+1}_head.pth")
        
        assert len(np.unique([len(pred_dict[k]) for k in pred_dict.keys()]))==1, f"the lengths of the lists are different: {[len(pred_dict[k]) for k in pred_dict.keys()]}"
        
        test_df = pd.DataFrame(pred_dict)
        test_df.to_csv(f"/mnt/bulk/timlenz/tumpe/RadiologyMLStation/radiologyml/data/07_model_output/test_preds_mace_{params['arch']}_f{fold+1}_head.csv",index=False)
        
        # Calculate AUROC
        test_auroc = roc_auc_score(all_targets, all_predicted_probs)

        pbar.close()

        tqdm.write(f"Test loss: {test_loss_avg:.5f}, Test Acc: {test_accuracy*100:.2f}, AUROC: {test_auroc:.4f}, bACC: {balanced_accuracy_score(all_targets, all_pred_labels)}")
