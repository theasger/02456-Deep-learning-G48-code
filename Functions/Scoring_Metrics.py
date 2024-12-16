import skimage
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
import torch


def DicePerClass(pred_labels,true_labels, num_classes=None):
    """
    Takes prediced labels and true label and computed a Dice-sørensen score for each class
    """
    if pred_labels.shape != true_labels.shape:
        raise ValueError("Predicted labels and True label must have the same dimensions")
    
    batchsize = true_labels.shape[0] if len(true_labels.shape) == 3 else 1
    num_classes = num_classes if num_classes else int(true_labels.max())

    DCS_scores= np.zeros((batchsize,num_classes))

    for b in range(batchsize):
        
        # pick single pair of images
        pred_label=pred_labels[b,:,:].copy() if batchsize != 1 else pred_labels.copy()
        true_label=true_labels[b,:,:].copy() if batchsize != 1 else true_labels.copy()

        for c in range(num_classes):
            # make binary images per class
            bin_pred=(pred_label == c)
            bin_true=(true_label == c)

            # only do calculation if class is present in either prediction or true label
            DCS_scores[b,c]= 1 - distance.dice(bin_pred.ravel(), bin_true.ravel()) if np.any(bin_true+bin_pred) else np.nan

    return DCS_scores


def dice_validation(model, val_dataloader, device, num_classes):
    """
    Takes a pytorch dataloader containing the validation data and the number of classes and returns a Numpy array containing dice scores
    for all classes in all images. 
    """
    model.to(device)
    all_scores=[]

    for X, Y in tqdm(val_dataloader, total=len(val_dataloader), leave=False):
        with torch.no_grad():
            X, Y = X.to(device), Y.to(device)
            Y_pred = model(X)
            Y_pred = torch.argmax(Y_pred, dim=1)

        ## Test function
        scores = DicePerClass(Y_pred.cpu().detach().numpy(),Y.cpu().detach().numpy(),num_classes=num_classes)

        all_scores.append(scores)

    all_scores=np.concatenate(all_scores)
    
    return all_scores

import torch
import torch.nn as nn
from torch.nn.functional import softmax

def flatten(input, target, ignore_index):
    num_class = input.size(1)
    input = input.permute(0, 2, 3, 1).contiguous()
    
    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)
    
    mask = (target_flatten != ignore_index)
    input_flatten = input_flatten[mask]
    target_flatten = target_flatten[mask]
    
    return input_flatten, target_flatten

class FocalTverskyLoss(nn.Module):
    """
    From https://github.com/YilmazKadir/Segmentation_Losses/blob/main/losses/focal_tversky.py
    """
    def __init__(self, smooth=1.0, alpha=0.5, beta=0.5, gamma=1):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        num_classes = input.size(1)
        input, target = input.flatten(start_dim=-2, end_dim=-1), target.flatten(start_dim=-2, end_dim=-1)
        input = softmax(input, dim=1)
        batch_losses = []
        

        for batch_idx in range(input.shape[0]):
            losses = []
            for c in range(num_classes):
                target_c = (target[batch_idx,:] == c).float()
                input_c = input[batch_idx, c,:]
                
                t_p = (input_c * target_c).sum()
                f_p = ((1-target_c) * input_c).sum()
                f_n = (target_c * (1-input_c)).sum()
                tversky = (t_p + self.smooth) / (t_p + self.alpha*f_p + self.beta*f_n + self.smooth)
                focal_tversky = (1 - tversky)**self.gamma
                losses.append(focal_tversky)
            
            batch_losses.append(torch.stack(losses).sum())
        batch_losses = torch.stack(batch_losses).mean()
        return batch_losses
    
