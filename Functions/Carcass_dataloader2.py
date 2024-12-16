# IMPORTS
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import random

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from skimage import io, color
import os
import numpy as np


# DATASET CLASS
class CarcassDataset(Dataset):

    def __init__(self, image_dir, label_dir, padding=True, padding_target=(960, 608), binary_class=False, foreground_only=False, transform=False):
        self.X_paths = [image_dir+'/'+image_name for image_name in os.listdir(image_dir)]
        self.X_paths=sorted(self.X_paths)  # Make sure images are in the right order
        self.y_paths = [label_dir+'/'+label_name for label_name in os.listdir(label_dir)]
        self.y_paths=sorted(self.y_paths)  # Make sure images are in the right order
        self.padding = padding
        self.padding_target = padding_target
        self.binary_class = binary_class
        self.foreground_only = foreground_only
        self.transform=transform

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, index):

        image = self._get_image(self.X_paths[index])
        label = self._get_mask(self.y_paths[index])

        #  Make padding for image and label
        if self.padding:
            image = self._padding(image, *self.padding_target)  # padding of colour image

        if self.padding:
            label = self._padding(label, *self.padding_target)

        # Apply options
        if self.foreground_only:
            image[label == 0, :] = 0  # background blacked out

        if self.binary_class:
            label[label != 0] = 1  # set all foreground classes to 1

        image = torch.Tensor(image)
        image = torch.movedim(image, 2, 0)
        label = torch.Tensor(label)

        if self.transform:   # Random rotation
            if random.random() > 0.4:
                angle = random.randint(-30, 30)
                image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR, expand=False)
                label = TF.rotate(label.unsqueeze(0), angle, interpolation=transforms.InterpolationMode.NEAREST, expand=False).squeeze(0)


            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

        return image, label

    def _get_image(self,path):
        image=io.imread(path)
        return image

    def _get_mask(self,path):
        mask=io.imread(path,as_gray=True)
        return mask

    def _padding(self, array, xx, yy):
        """
        :param array: numpy array
        :param xx: desired height
        :param yy: desired width
        :return: padded array
        """

        h = array.shape[0]
        w = array.shape[1]

        a = (xx - h) // 2
        aa = xx - a - h

        b = (yy - w) // 2
        bb = yy - b - w

        if len(array.shape) == 3:
            color = np.median(array, axis=(0,1))
            img = np.stack([np.pad(array[:,:,c], pad_width=((a, aa), (b, bb)), mode='constant', constant_values=color[c]) for c in range(3)], axis=2)
            return img
        else:
            return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

