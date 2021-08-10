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

    def load_data_dir(self):
        image, label, mask = [], [], []
        mode = self.mode

        img_dir = os.path.join(self.dataset_path, self.category, mode)
        gt_dir = os.path.join(self.dataset_path, self.category, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            image.extend(img_fpath_list)

            if img_type == 'good':
                label.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                label.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                
        return list(image), list(label), list(mask)
