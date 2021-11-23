# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import datasets
#import opendatasets as od
import os
from random import randint
import urllib
import zipfile
import copy
import torch.nn.functional as F
import wget

# Setup function to create dataloaders for image datasets
def generate_dataloader(data, name,batch_size,use_cuda, transform):
    if data is None:
        return None

    # Read image files to pytorch dataset using ImageFolder, a generic data
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}

    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(name == "train"),
                            **kwargs)

    return dataloader


def setup_TinyImagenet(DATA_DIR,TRAIN_DIR,VALID_DIR):
    # Retrieve data directly from Stanford data source
    wget.download('http://cs231n.stanford.edu/tiny-imagenet-200.zip')

    # Unzip raw zip file
    with zipfile.ZipFile('tiny-imagenet-200.zip', 'r') as zip_ref:
        zip_ref.extractall('tiny-imagenet-200')

    # Create separate validation subfolders for the validation images based on
    # their labels indicated in the val_annotations txt file
    val_img_dir = os.path.join(VALID_DIR, 'images')

    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create subfolders (if not present) for validation images based on label,
    # and move images into the respective folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

    return val_img_dir

def train_with_tensorboard(dataloader,model,loss_fn,optimizer,writer,epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0
    running_correct = 0
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        #Compute the prediction error
        pred = model(X)
        loss = loss_fn(pred,y)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += (pred.argmax(1) == y).type(torch.float).sum().item()/dataloader.batch_size
        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print("loss: "+ str(loss) + " " +  str(current/size) )

            writer.add_scalar('training loss',running_loss/100,epoch*len(dataloader)+batch)
            writer.add_scalar('training acc', running_correct / 100, epoch * len(dataloader) + batch)
            running_loss = 0.0
            running_correct = 0.0


def test_with_tensorboard(dataloader, model,loss_fn,writer,epoch,training_b_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss/=num_batches
    correct/=size
    writer.add_scalar('validation loss', test_loss, epoch * training_b_size)
    writer.add_scalar('validation acc', correct, epoch * training_b_size)

    print("Test Error: \n" + "Accuracy: " + str(100*correct) + "Avg loss:" +str(test_loss))