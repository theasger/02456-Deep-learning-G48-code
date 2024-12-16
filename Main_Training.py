#!/zhome/7c/0/155777/anaconda3/envs/ml/bin/python

### imports ###
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import date
from skimage import io, color
from skimage.color import label2rgb
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.utils.data import Subset
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import gc

import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

# Custom class
from Functions.Carcass_dataloader2 import *
from Functions.Carcass_trainer import *
from Functions.CustomNets import *
from Functions.Scoring_Metrics import *

from argparse import ArgumentParser

### functions ###
def train_val_dataset(dataset, val_split=0.25,random_state=42):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=random_state)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

if __name__ == "__main__":
    

    # Train part
    parser = ArgumentParser(description="Training script for segmentation models")
    # Data paths
    parser.add_argument("-image", action="store", dest="image_dir_name", type=str, default='images_correct_labels' , help="Directory containing images")
    parser.add_argument("-label", action="store", dest="label_dir_name", type=str, default='mask_images_correct_labels' , help="Directory containing image labels")
    
    model_list=["fcn",
                "unet",
                "unet_pretrained",
                "dlv3",
                "dlv3p"]
    
    parser.add_argument("-model", action="store", dest="model_pick", type=str, default='dlv3',
                        help=f'Chose a model type from these; {model_list}')


    loss_list=["FL",
               "TL",
               "CE"]

    parser.add_argument("-l", action="store", dest="loss", type=str, default='FL',
                        help=f'Chose a loss type from these; {loss_list}')

    parser.add_argument("-ne", action="store", dest="n_epochs", type=int, default=300, help="Number of epochs")

    args = parser.parse_args()
    image_dir_name = args.image_dir_name
    label_dir_name = args.label_dir_name
    model_pick = args.model_pick
    loss = args.loss
    n_epochs = args.n_epochs
    

    #### Training ####
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Setup directories
    data_dir = os.path.join('/zhome/7c/0/155777/Desktop/Code/Data')
    output_dir = os.path.join(data_dir, f"output_{loss.lower()}")
    image_dir = os.path.join(data_dir, image_dir_name)
    label_dir = os.path.join(data_dir, label_dir_name)
    
    # define number of classes in images
    num_classes=6

    if model_pick.lower() == 'fcn':
        model_name="FCN_custom_sorted_classes"
        model=FCN(num_classes)

    elif model_pick.lower() == 'unet':
        # Setup model
        model_name="UNet_custom_sorted_classes"
        model = UNet(num_classes)

    elif model_pick.lower() == 'unet_pretrained':
        # Setup model
        model_name="UNet_resnet50_pretrained_sorted_classes"
        model = smp.Unet('resnet50',in_channels=3, encoder_weights='imagenet',classes=num_classes, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

    elif model_pick.lower() == 'dlv3':
        # Setup model
        model_name="deeplab_v3_resnet50_pretrained_sorted_classes"
        model = smp.DeepLabV3('resnet50',in_channels=3, encoder_weights='imagenet',classes=num_classes)
    
    elif model_pick.lower() == 'dlv3p':
        # Setup model
        model_name="DeepLabV3Plus_resnet50_pretrained_sorted_classes"
        model = smp.DeepLabV3Plus('resnet50',in_channels=3, encoder_weights='imagenet',classes=num_classes)

    else:
        raise ValueError(f"{model_pick} in not not in {model_list}")


    if loss.lower() == "fl":
        criterion = smp.losses.FocalLoss(mode="multiclass", alpha=0.7, gamma=2)
        class_weights = None
    elif loss.lower() == "tl":
        criterion = smp.losses.TverskyLoss(mode="multiclass", alpha=0.4, smooth=0.6)
        class_weights = None
    elif loss.lower() == "ce":
        criterion = nn.CrossEntropyLoss
        class_weights = torch.tensor([1.0,1.0,3.0,2.0,2.0,3.0])    
    else:
        raise ValueError(f"{loss} in not not in {loss_list}")

    optimizer = optim.Adam
    dataset = CarcassDataset(image_dir, label_dir,transform=True)
    split_datasets = train_val_dataset(dataset)
    data_loader = DataLoader
    lr=0.0001
    batch_size=5


    args = (model,
            criterion,
            optimizer,
            split_datasets['train'],
            data_loader,
            device,
            lr,
            n_epochs,
            batch_size,
            class_weights)

    trainer = Trainer(*args)

    # start training
    step_losses, epoch_losses = trainer.train(n_epochs,plot=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[0].set_title("Iteration training Loss")
    axes[1].plot(epoch_losses)
    axes[1].set_title("Epoch average training Loss")
    fig.savefig(f"{output_dir}/{model_name}_losses.png")

    model_filename = f"{model_name}_{date.today()}.pth"
    torch.save(model.state_dict(), output_dir+'/'+model_filename)  # Save the trained model

    #### Testing ####

    # Test with dice function on all test data
    split_datasets["val"].dataset.transform=False

    data_loader_test = DataLoader(split_datasets["val"], batch_size=batch_size)

    all_scores=dice_validation(model, data_loader_test, device, num_classes)  # run test from Scoring Metrics
    mean_scores=np.nanmean(all_scores,axis=0)

    np.save(output_dir+f"/{model_name}_Raw_dice_{date.today()}",all_scores)  # save all scores to np.array-file

    with open(output_dir+f"/{model_name}_dice_results_{date.today()}.txt","w") as f:  # Save mean scores per class to a .txt
        for c in range(num_classes):
            f.write(f"Class_{c}:\t{mean_scores[c]}\n")
