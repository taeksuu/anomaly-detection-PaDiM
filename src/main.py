from Dataset import MVTecADDataset, CATEGORIES

import torch
from torchvision.models import resnet18, resnet152, wide_resnet50_2, wide_resnet101_2, resnext101_32x8d
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
import random
from random import sample
import os
import matplotlib.pyplot as plt
plt.ion()
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_directory', type=str, default='D:/mvtec_anomaly_detection')
    parser.add_argument('--save_directory', type=str, default='./PaDiM_results')
    parser.add_argument('--backbone_model', type=str, choices=['resnet18', 'resnet152', 'wide_resnet50_2', 'wide_resnet101_2', 'resnext101_32x8d'], default='wide_resnet50_2')
    return parser.parse_args(args=[])
  
  
args = parse_args()

if args.backbone_model == 'resnet18':
    model = resnet18(pretrained=True, progress=True)
    embedded_vector_size = 448
    reduced_vector_size = 100
    
elif args.backbone_model == 'resnet152':
    model = resnet152(pretrained=True, progress=True)
    embedded_vector_size = 1792
    reduced_vector_size = 550
    
    
elif args.backbone_model == 'wide_resnet50_2':
    model = wide_resnet50_2(pretrained=True, progress=True)
    embedded_vector_size = 1792
    reduced_vector_size = 550
    
    
elif args.backbone_model == 'wide_resnet101_2':
    model = wide_resnet101_2(pretrained=True, progress=True)
    embedded_vector_size = 1792
    reduced_vector_size = 550
    
    
elif args.backbone_model == 'resnext101_32x8d':
    model = resnext101_32x8d(pretrained=True, progress=True)
    embedded_vector_size = 1792
    reduced_vector_size = 550
    
model.to(device)
model.eval()

seed = 20210809
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available: torch.cuda.manual_seed_all(seed)
    
#for randomly reducing embedding vector size
idx = torch.tensor(sample(range(0, embedded_vector_size), reduced_vector_size))

outputs = []

def hook(module, input, output):
    outputs.append(output)
    
model.layer1[-1].register_forward_hook(hook)
model.layer2[-1].register_forward_hook(hook)
model.layer3[-1].register_forward_hook(hook)

os.makedirs(os.path.join(args.save_directory, "temp_{}".format(args.backbone_model)), exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
fig_img_rocauc = ax[0]
fig_pixel_rocauc = ax[1]

total_img_rocauc = []
total_pixel_rocauc = []

for category in CATEGORIES:
    train_dataset = MVTecADDataset(dataset_path=args.dataset_directory, mode="train", category=category, transform=True)
    test_dataset = MVTecADDataset(dataset_path=args.dataset_directory, mode="test", category=category, transform=True)

    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
    
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    train_feature_save_directory = os.path.join(args.save_directory, "temp_{}".format(args.backbone_model), "train_{}.pkl".format(category))
    if not os.path.exists(train_feature_save_directory):
        for (x, _, _) in tqdm(train_dataloader, '| Extracting features from train data | {} |'.format(category)):
            with torch.no_grad():
                _ = model(x.to(device))
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)
            
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
  test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

  train_feature_save_directory = os.path.join(args.save_directory, "temp_{}".format(args.backbone_model), "train_{}.pkl".format(category))
  
  if not os.path.exists(train_feature_save_directory):
      for (x, _, _) in tqdm(train_dataloader, '| Extracting features from train data | {} |'.format(category)):
          #input train image into pretrained backbone model
          with torch.no_grad():
              _ = model(x.to(device))
          for k, v in zip(train_outputs.keys(), outputs):
              train_outputs[k].append(v.cpu().detach())
          outputs = []
      for k, v in train_outputs.items():
          train_outputs[k] = torch.cat(v, 0)

      #concatenate torch.Size([dataset_size, 256, 56, 56]), torch.Size([dataset_size, 512, 28, 28]), torch.Size([dataset_size, 1024, 14, 14]) into torch.Size([dataset_size, 1792, 56, 56])
      embedded_vectors = concat_embedding_vectors(train_outputs['layer1'], train_outputs['layer2'])
      embedded_vectors = concat_embedding_vectors(embedded_vectors, train_outputs['layer2'])

      #randomly reduce embedded vector size to torch.Size([dataset_size, 550, 56, 56])
      embedded_vectors = torch.index_select(embedded_vectors, 1, idx)

      #calculate gaussian distribution
      B, C, H, W = embedded_vectors.size()
      embedded_vectors = embedded_vectors.view(B, C, H * W)

      mean = torch.mean(embedded_vectors, dim=0).numpy()
      cov = torch.zeros(C, C, H * W).numpy()
      I = np.identity(C)
      for i in range(H * W):
          cov[:, :, i] = np.cov(embedded_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I

      #save learned distribution
      train_outputs = [mean, cov]
      with open(train_feature_save_directory, 'wb') as f:
          pickle.dump(train_outputs, f)
  else:
      with open(train_feature_save_directory, 'rb') as f:
          train_outputs = pickle.load(f)
