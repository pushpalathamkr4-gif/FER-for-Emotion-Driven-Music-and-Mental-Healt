"""
=========================================================
FER2013 Dataset Loader

Author : Your Name
=========================================================
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms


#############################################################
# FER Dataset
#############################################################

class FER2013Dataset(Dataset):

    def __init__(self,
                 csv_file,
                 usage="Training",
                 transform=None):

        self.data = pd.read_csv(csv_file)

        self.data = self.data[self.data["Usage"] == usage]

        self.transform = transform

    #########################################################

    def __len__(self):

        return len(self.data)

    #########################################################

    def __getitem__(self, index):

        row = self.data.iloc[index]

        emotion = int(row["emotion"])

        pixels = np.asarray(
            row["pixels"].split(),
            dtype=np.uint8
        )

        image = pixels.reshape(48,48)

        image = Image.fromarray(image)

        image = image.convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, emotion


#############################################################
# Data Loader Class
#############################################################

class FERDataLoader:

    def __init__(self,
                 csv_path,
                 batch_size=32):

        self.csv_path = csv_path

        self.batch_size = batch_size

        #####################################################

        self.train_transform = transforms.Compose([

            transforms.Resize((224,224)),

            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(10),

            transforms.ColorJitter(

                brightness=0.2,
                contrast=0.2

            ),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[0.485,0.456,0.406],

                std=[0.229,0.224,0.225]

            )

        ])

        #####################################################

        self.test_transform = transforms.Compose([

            transforms.Resize((224,224)),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[0.485,0.456,0.406],

                std=[0.229,0.224,0.225]

            )

        ])

    #########################################################

    def train_loader(self):

        dataset = FER2013Dataset(

            self.csv_path,

            usage="Training",

            transform=self.train_transform

        )

        loader = DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=True,

            num_workers=4,

            pin_memory=True

        )

        return loader

    #########################################################

    def validation_loader(self):

        dataset = FER2013Dataset(

            self.csv_path,

            usage="PublicTest",

            transform=self.test_transform

        )

        loader = DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=False,

            num_workers=4

        )

        return loader

    #########################################################

    def test_loader(self):

        dataset = FER2013Dataset(

            self.csv_path,

            usage="PrivateTest",

            transform=self.test_transform

        )

        loader = DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=False,

            num_workers=4

        )

        return loader
