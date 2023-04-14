import random
import pandas as pd
import numpy as np
import os
import glob
import argparse
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

import torchvision.models as models
import torchvision.transforms as transforms
from timm.loss import LabelSmoothingCrossEntropy

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from dataset import WallPaper, class_name, get_data_list
from models import BaseModel
from transforms import get_transforms
from sampler import ImbalancedDatasetSampler

import warnings
warnings.filterwarnings(action='ignore') 


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed = 41
seed_everything(seed) # Seed 고정


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, optimizer, train_loader, val_loader, scheduler, args, start_epoch=1):
    wandb.init(
        project=args.exp,

        config = {
            "lr" : args.lr,
            "batch_size": args.batch_size,
            "epoch": args.epoch
        }
    )
    model.to(device)
    criterion = LabelSmoothingCrossEntropy().to(device)
    
    best_score = 0
    best_model = None
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epoch+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = model(imgs)
                loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')
        wandb.log({
            "train_loss": _train_loss,
            "val_loss": _val_loss,
            "val_score": _val_score
        })
       
        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_score < _val_score:
            best_score = _val_score
            best_model = model
            state = {
                'net': best_model.state_dict(),
                'f1': best_score,
                'epoch': epoch,
            }
            print(f'Save {epoch} epoch model (F1 : {best_score}) to best model.')
            os.makedirs(os.path.join(args.exp, "checkpoint"), exist_ok=True)
            torch.save(state, os.path.join(args.exp, "checkpoint", "best_model.pth"))
        
        if epoch == args.epoch:
            with open(os.path.join(args.exp, "performance.txt"), "w") as f:
                f.write(f"Epoch: {state['epoch']}, F1: {state['f1']} \n")
            f.close()
    return best_model


def validation(model, criterion, val_loader):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='weighted')
    
    return _val_loss, _val_score


def inference(model, test_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = [class_name[pred] for pred in preds]
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wallpaper defects classification')
    parser.add_argument("--exp", type=str,
                        help="name of experiment to store logs and models")
    parser.add_argument("--data_dir", type=str, default="/workspace/dataset/dacon_wallpaper_defects",
                        help="dataset folder")
    parser.add_argument("-lr", type=float, default=5e-2,
                        help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="num workers")
    parser.add_argument("--epoch", type=int, default=100,
                        help="num workers")
    parser.add_argument("--infer_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.exp, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model = BaseModel()
    model.eval()

    tr, val, ts = get_data_list(args.data_dir, seed=seed)

    train_dataset = WallPaper(tr['img_path'].values, tr['label'].values, transforms=get_transforms(True)) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), 
                              num_workers=args.num_workers, pin_memory=True)
    
    val_dataset = WallPaper(val['img_path'].values, val['label'].values, transforms=get_transforms(False))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    test_dataset = WallPaper(ts, None, transforms=get_transforms(False))
    test_loader = DataLoader(test_dataset, batch_size=8, pin_memory=True)
    
    optimizer = torch.optim.SGD(params = model.parameters(), lr = args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                optimizer, T_0=50, T_mult=2, eta_min=0)

    if not args.infer_only:
        start_epoch = 1
        if args.resume:
            checkpoint = torch.load(os.path.join(args.exp, "checkpoint", "best_model.pth"))
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["net"])
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, args, start_epoch)
    else:
        checkpoint = torch.load(os.path.join(args.exp, "checkpoint", "best_model.pth"))
        model.load_state_dict(checkpoint["net"])
        infer_model = model.to(device)
    preds = inference(infer_model, test_loader)
    submit = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    submit['label'] = preds
    submit.to_csv(os.path.join("results", str(args.exp) + '_submit.csv'), index=False)
    wandb.finish()