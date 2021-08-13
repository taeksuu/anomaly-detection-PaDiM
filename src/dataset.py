import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random


# In[4]:


CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut','leather', 'metal_nut', 
            'pill', 'screw', 'tile', 'toothbrush','transistor', 'wood', 'zipper']


# In[163]:


class MVTecADDataset(Dataset):
    def __init__(self, dataset_path, mode, category, transform=False):
        assert category in CATEGORIES, "category: {} should be in {}".format(category, CATEGORIES)
        self.dataset_path = dataset_path
        self.mode = mode
        self.category = category
        self.transform = transform

        # seed = np.random.randint(20210809)
        self.transform_image = transforms.Compose([
            # transforms.Resize(256),
            # transforms.ToTensor()
            # transforms.RandomAffine(30, translate=(0.2, 0.2)),
            transforms.Resize(256, Image.ANTIALIAS),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose([
            # transforms.Resize(256),
            # transforms.ToTensor()
            # transforms.RandomAffine(30, translate=(0.2, 0.2)),
            transforms.Resize(256, Image.NEAREST),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        self.image, self.label, self.mask = self.load_data_dir()

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image, label, mask = self.image[idx], self.label[idx], self.mask[idx]

        image = Image.open(image).convert('RGB')
        if self.transform:
            # random.seed(seed)
            image = self.transform_image(image)

        if label == 0:
            mask = torch.zeros([1, 224, 224])
        else:
            mask = Image.open(mask)
            if self.transform:
                # random.seed(seed)
                mask = self.transform_mask(mask)

        return image, label, mask

    
