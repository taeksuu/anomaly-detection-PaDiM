from dataset import MVTecADDataset, CATEGORIES

import torch
from torchvision.models import resnet18, resnet152, wide_resnet50_2, wide_resnet101_2, resnext101_32x8d
from torch.utils.data import DataLoader
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

import argparse
import random
from random import sample
import os
import matplotlib.pyplot as plt
import matplotlib

from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import pickle
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import morphology
from skimage.segmentation import mark_boundaries

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


# In[2]:


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_directory', type=str, default='D:/mvtec_anomaly_detection')
    parser.add_argument('--save_directory', type=str, default='D:/PaDiM_results')
    parser.add_argument('--backbone_model', type=str,
                        choices=['resnet18', 'resnet152', 'wide_resnet50_2', 'wide_resnet101_2', 'resnext101_32x8d',
                                 'efficientnet_b5', 'efficientnet_b7'], default='wide_resnet50_2')
    parser.add_argument('--layer_used', type=str, choices=['1', '2', '3', '1+2+3'], default='1+2+3')
    parser.add_argument('--dimensionality_reduction', type=int, default=-1)
    parser.add_argument('--gaussian_regularisation_term', type=float, default=0.01)
    parser.add_argument('--interpolation_mode', type=str, choices=['bicubic', 'bilinear'], default='bicubic')
    parser.add_argument('--gaussian_filter_std', type=float, default=4.0)
    return parser.parse_args()


# In[5]:


def main():
    args = parse_args()

    if args.backbone_model == 'resnet18':
        model = resnet18(pretrained=True, progress=True)

        layer_1_size = 64
        layer_2_size = 128
        layer_3_size = 256

        if args.layer_used == '1':
            embedded_vector_size = layer_1_size
        elif args.layer_used == '2':
            embedded_vector_size = layer_2_size
        elif args.layer_used == '3':
            embedded_vector_size = layer_3_size
        elif args.layer_used == '1+2+3':
            embedded_vector_size = layer_1_size + layer_2_size + layer_3_size

        reduced_vector_size = args.dimensionality_reduction
        if reduced_vector_size == -1:
            reduced_vector_size = embedded_vector_size

        if embedded_vector_size < reduced_vector_size:
            print("Reduced dimension should be less than or equal to {}".format(embedded_vector_size))
        reduced_layer_1_size, reduced_layer_2_size, reduced_layer_3_size = get_reduced_vector_size_per_layers(
            reduced_vector_size, layer_1_size, layer_2_size, layer_3_size)

    elif args.backbone_model == 'resnet152':
        model = resnet152(pretrained=True, progress=True)

        layer_1_size = 256
        layer_2_size = 512
        layer_3_size = 1024

        if args.layer_used == '1':
            embedded_vector_size = layer_1_size
        elif args.layer_used == '2':
            embedded_vector_size = layer_2_size
        elif args.layer_used == '3':
            embedded_vector_size = layer_3_size
        elif args.layer_used == '1+2+3':
            embedded_vector_size = layer_1_size + layer_2_size + layer_3_size

        reduced_vector_size = args.dimensionality_reduction
        if reduced_vector_size == -1:
            reduced_vector_size = embedded_vector_size

        if embedded_vector_size < reduced_vector_size:
            print("Reduced dimension should be less than or equal to {}".format(embedded_vector_size))
        reduced_layer_1_size, reduced_layer_2_size, reduced_layer_3_size = get_reduced_vector_size_per_layers(
            reduced_vector_size, layer_1_size, layer_2_size, layer_3_size)

    elif args.backbone_model == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)

        layer_1_size = 256
        layer_2_size = 512
        layer_3_size = 1024

        if args.layer_used == '1':
            embedded_vector_size = layer_1_size
        elif args.layer_used == '2':
            embedded_vector_size = layer_2_size
        elif args.layer_used == '3':
            embedded_vector_size = layer_3_size
        elif args.layer_used == '1+2+3':
            embedded_vector_size = layer_1_size + layer_2_size + layer_3_size

        reduced_vector_size = args.dimensionality_reduction
        if reduced_vector_size == -1:
            reduced_vector_size = embedded_vector_size

        if embedded_vector_size < reduced_vector_size:
            print("Reduced dimension should be less than or equal to {}".format(embedded_vector_size))
        reduced_layer_1_size, reduced_layer_2_size, reduced_layer_3_size = get_reduced_vector_size_per_layers(
            reduced_vector_size, layer_1_size, layer_2_size, layer_3_size)

    elif args.backbone_model == 'wide_resnet101_2':
        model = wide_resnet101_2(pretrained=True, progress=True)

        layer_1_size = 256
        layer_2_size = 512
        layer_3_size = 1024

        if args.layer_used == '1':
            embedded_vector_size = layer_1_size
        elif args.layer_used == '2':
            embedded_vector_size = layer_2_size
        elif args.layer_used == '3':
            embedded_vector_size = layer_3_size
        elif args.layer_used == '1+2+3':
            embedded_vector_size = layer_1_size + layer_2_size + layer_3_size

        reduced_vector_size = args.dimensionality_reduction
        if reduced_vector_size == -1:
            reduced_vector_size = embedded_vector_size

        if embedded_vector_size < reduced_vector_size:
            print("Reduced dimension should be less than or equal to {}".format(embedded_vector_size))
        reduced_layer_1_size, reduced_layer_2_size, reduced_layer_3_size = get_reduced_vector_size_per_layers(
            reduced_vector_size, layer_1_size, layer_2_size, layer_3_size)

    elif args.backbone_model == 'resnext101_32x8d':
        model = resnext101_32x8d(pretrained=True, progress=True)

        layer_1_size = 256
        layer_2_size = 512
        layer_3_size = 1024

        if args.layer_used == '1':
            embedded_vector_size = layer_1_size
        elif args.layer_used == '2':
            embedded_vector_size = layer_2_size
        elif args.layer_used == '3':
            embedded_vector_size = layer_3_size
        elif args.layer_used == '1+2+3':
            embedded_vector_size = layer_1_size + layer_2_size + layer_3_size

        reduced_vector_size = args.dimensionality_reduction
        if reduced_vector_size == -1:
            reduced_vector_size = embedded_vector_size

        if embedded_vector_size < reduced_vector_size:
            print("Reduced dimension should be less than or equal to {}".format(embedded_vector_size))
        reduced_layer_1_size, reduced_layer_2_size, reduced_layer_3_size = get_reduced_vector_size_per_layers(
            reduced_vector_size, layer_1_size, layer_2_size, layer_3_size)

    elif args.backbone_model == 'efficientnet_b5':
        model = EfficientNet.from_pretrained('efficientnet-b5')

        layer_1_size = 40
        layer_2_size = 176
        layer_3_size = 304

        if args.layer_used == '1':
            embedded_vector_size = layer_1_size
        elif args.layer_used == '2':
            embedded_vector_size = layer_2_size
        elif args.layer_used == '3':
            embedded_vector_size = layer_3_size
        elif args.layer_used == '1+2+3':
            embedded_vector_size = layer_1_size + layer_2_size + layer_3_size

        reduced_vector_size = args.dimensionality_reduction
        if reduced_vector_size == -1:
            reduced_vector_size = embedded_vector_size

        if embedded_vector_size < reduced_vector_size:
            print("Reduced dimension should be less than or equal to {}".format(embedded_vector_size))
        reduced_layer_1_size, reduced_layer_2_size, reduced_layer_3_size = get_reduced_vector_size_per_layers(
            reduced_vector_size, layer_1_size, layer_2_size, layer_3_size)

    elif args.backbone_model == 'efficientnet_b7':
        model = EfficientNet.from_pretrained('efficientnet-b7')

        layer_1_size = 48
        layer_2_size = 160
        layer_3_size = 224

        if args.layer_used == '1':
            embedded_vector_size = layer_1_size
        elif args.layer_used == '2':
            embedded_vector_size = layer_2_size
        elif args.layer_used == '3':
            embedded_vector_size = layer_3_size
        elif args.layer_used == '1+2+3':
            embedded_vector_size = layer_1_size + layer_2_size + layer_3_size

        reduced_vector_size = args.dimensionality_reduction
        if reduced_vector_size == -1:
            reduced_vector_size = embedded_vector_size

        if embedded_vector_size < reduced_vector_size:
            print("Reduced dimension should be less than or equal to {}".format(embedded_vector_size))
        reduced_layer_1_size, reduced_layer_2_size, reduced_layer_3_size = get_reduced_vector_size_per_layers(
            reduced_vector_size, layer_1_size, layer_2_size, layer_3_size)

    model.to(device)
    model.eval()

    # control randomness
    seed = 20210809
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available: torch.cuda.manual_seed_all(seed)

    # for randomly reducing embedding vector size
    if args.dimensionality_reduction != -1:
        if args.layer_used == '1':
            layer_1_idx = torch.tensor(sample(range(0, layer_1_size), reduced_vector_size))
        elif args.layer_used == '2':
            layer_2_idx = torch.tensor(sample(range(0, layer_2_size), reduced_vector_size))
        elif args.layer_used == '3':
            layer_3_idx = torch.tensor(sample(range(0, layer_3_size), reduced_vector_size))
        elif args.layer_used == '1+2+3':
            layer_1_idx = torch.tensor(sample(range(0, layer_1_size), reduced_layer_1_size))
            layer_2_idx = torch.tensor(sample(range(0, layer_2_size), reduced_layer_2_size))
            layer_3_idx = torch.tensor(sample(range(0, layer_3_size), reduced_layer_3_size))
    else:
        if args.layer_used == '1':
            layer_1_idx = torch.tensor(sample(range(0, layer_1_size), reduced_vector_size))
        elif args.layer_used == '2':
            layer_2_idx = torch.tensor(sample(range(0, layer_2_size), reduced_vector_size))
        elif args.layer_used == '3':
            layer_3_idx = torch.tensor(sample(range(0, layer_3_size), reduced_vector_size))
        elif args.layer_used == '1+2+3':
            layer_1_idx = torch.tensor(sample(range(0, layer_1_size), layer_1_size))
            layer_2_idx = torch.tensor(sample(range(0, layer_2_size), layer_2_size))
            layer_3_idx = torch.tensor(sample(range(0, layer_3_size), layer_3_size))

    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    if args.backbone_model == 'efficientnet_b5' or args.backbone_model == 'efficientnet_b7':
        model._blocks[7].register_forward_hook(hook)
        model._blocks[20].register_forward_hook(hook)
        model._blocks[28].register_forward_hook(hook)
    else:
        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_directory, "{}_Layer_{}_Rd_{}".format(args.backbone_model, args.layer_used,
                                                                                  args.dimensionality_reduction)),
                exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_img_rocauc = []
    total_pixel_rocauc = []

    for category in CATEGORIES:
        train_dataset = MVTecADDataset(dataset_path=args.dataset_directory, mode="train", category=category,
                                       transform=True)
        test_dataset = MVTecADDataset(dataset_path=args.dataset_directory, mode="test", category=category,
                                      transform=True)

        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract features from train data
        os.makedirs(os.path.join(args.save_directory,
                                 "temp_{}_Layer_{}_Rd_{}".format(args.backbone_model, args.layer_used,
                                                                 args.dimensionality_reduction), category),
                    exist_ok=True)
        train_feature_save_directory = os.path.join(args.save_directory,
                                                    "temp_{}_Layer_{}_Rd_{}".format(args.backbone_model,
                                                                                    args.layer_used,
                                                                                    args.dimensionality_reduction,
                                                                                    category), category,
                                                    "train_{}.pkl".format(category))
        if not os.path.exists(train_feature_save_directory):
            for (image, _, _) in tqdm(train_dataloader,
                                      '| Extracting features from train data | {} |'.format(category)):
                # input train image into pretrained backbone model
                with torch.no_grad():
                    _ = model(image.to(device))
                # save output of each layer to OrderedDict and clear outputs for next batch
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                outputs = []
            # concatenate batch data
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            if args.layer_used == '1':
                # randomly reduce output vector size of layer1 to designated size
                embedded_vectors = torch.index_select(train_outputs['layer1'], 1, layer_1_idx)
            elif args.layer_used == '2':
                # randomly reduce output vector size of layer2 to designated size
                embedded_vectors = torch.index_select(train_outputs['layer2'], 1, layer_2_idx)
            elif args.layer_used == '3':
                # randomly reduce output vector size of layer3 to designated size
                embedded_vectors = torch.index_select(train_outputs['layer3'], 1, layer_3_idx)
            elif args.layer_used == '1+2+3':
                # randomly reduce output vector size of each layer to designated size
                reduced_layer_1 = torch.index_select(train_outputs['layer1'], 1, layer_1_idx)
                reduced_layer_2 = torch.index_select(train_outputs['layer2'], 1, layer_2_idx)
                reduced_layer_3 = torch.index_select(train_outputs['layer3'], 1, layer_3_idx)

                # concatenate reduced_layer_1, reduced_layer_2 and reduced_layer_3
                embedded_vectors = concat_embedding_vectors(reduced_layer_1, reduced_layer_2)
                embedded_vectors = concat_embedding_vectors(embedded_vectors, reduced_layer_3)

            # calculate gaussian parameters
            B, C, H, W = embedded_vectors.size()
            embedded_vectors = embedded_vectors.view(B, C, H * W)

            mean = torch.mean(embedded_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in tqdm(range(H * W), '| Calculating gaussian parameters |'):
                cov[:, :, i] = np.cov(embedded_vectors[:, :, i].numpy(),
                                      rowvar=False) + args.gaussian_regularisation_term * I

            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_save_directory, 'wb') as f:
                pickle.dump(train_outputs, f)
                print("Succesfully saved Gaussian parameters")
        # load extracted features from train data if exists
        else:
            with open(train_feature_save_directory, 'rb') as f:
                train_outputs = pickle.load(f)
                print("Succesfully loaded Gaussian parameters")

        test_img_list = []
        test_label_list = []
        test_mask_list = []

        # extract features from test data
        for (image, label, mask) in tqdm(test_dataloader,
                                         '| Extracting features from test data | {} |'.format(category)):
            test_img_list.extend(image.cpu().detach().numpy())
            test_label_list.extend(label.cpu().detach().numpy())
            test_mask_list.extend(mask.cpu().detach().numpy())

            # input test image into pretrained backbone model
            with torch.no_grad():
                _ = model(image.to(device))
            # save output of each layer to OrderedDict and clear outputs for next batch
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            outputs = []
        # concatenate batch data
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        if args.layer_used == '1':
            # randomly reduce output vector size of layer1 to designated size
            embedded_vectors = torch.index_select(test_outputs['layer1'], 1, layer_1_idx)
        elif args.layer_used == '2':
            # randomly reduce output vector size of layer2 to designated size
            embedded_vectors = torch.index_select(test_outputs['layer2'], 1, layer_2_idx)
        elif args.layer_used == '3':
            # randomly reduce output vector size of layer3 to designated size
            embedded_vectors = torch.index_select(test_outputs['layer3'], 1, layer_3_idx)
        elif args.layer_used == '1+2+3':
            # randomly reduce output vector size of each layer to designated size
            reduced_layer_1 = torch.index_select(test_outputs['layer1'], 1, layer_1_idx)
            reduced_layer_2 = torch.index_select(test_outputs['layer2'], 1, layer_2_idx)
            reduced_layer_3 = torch.index_select(test_outputs['layer3'], 1, layer_3_idx)

            # concatenate reduced_layer_1, reduced_layer_2 and reduced_layer_3
            embedded_vectors = concat_embedding_vectors(reduced_layer_1, reduced_layer_2)
            embedded_vectors = concat_embedding_vectors(embedded_vectors, reduced_layer_3)

        # calculate distance
        B, C, H, W = embedded_vectors.size()
        embedded_vectors = embedded_vectors.view(B, C, H * W).numpy()

        distances = []
        for i in tqdm(range(H * W), '| Calculating Mahalanobis distance |'):
            mean = train_outputs[0][:, i]
            cov = train_outputs[1][:, :, i]
            cov_inv = np.linalg.inv(cov)
            distance = [mahalanobis(u=vector[:, i], v=mean, VI=cov_inv) for vector in embedded_vectors]
            distances.append(distance)
        distances = np.array(distances).transpose(1, 0).reshape(B, H, W)

        # resize distances(n x H * W) to (n x 224 * 224) by interpolation of given mode for anomaly map
        distances = torch.tensor(distances)
        anomaly_map = F.interpolate(distances.unsqueeze(1), size=image.size(2), mode=args.interpolation_mode,
                                    align_corners=False).squeeze().numpy()

        # apply Gaussian filter
        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=args.gaussian_filter_std)

        # normalization to 0 - 1
        max_score = anomaly_map.max()
        min_score = anomaly_map.min()
        normalized_scores = (anomaly_map - min_score) / (max_score - min_score)

        # image-level ROCAUC
        img_scores = normalized_scores.reshape(normalized_scores.shape[0], -1).max(axis=1)
        test_label_list = np.asarray(test_label_list)
        FPR, TPR, _ = roc_curve(test_label_list, img_scores)
        img_rocauc = roc_auc_score(test_label_list, img_scores)
        total_img_rocauc.append(img_rocauc)
        print("image-level ROCAUC: {}".format(img_rocauc))
        fig_img_rocauc.plot(FPR, TPR, label="{} image-level ROCAUC: {}".format(category, img_rocauc))

        # pixel-level ROCAUC
        test_mask_list = np.asarray(test_mask_list)
        FPR, TPR, _ = roc_curve(test_mask_list.flatten(), normalized_scores.flatten())
        pixel_rocauc = roc_auc_score(test_mask_list.flatten(), normalized_scores.flatten())
        total_pixel_rocauc.append(pixel_rocauc)
        print("pixel-level ROCAUC: {}".format(pixel_rocauc))
        fig_pixel_rocauc.plot(FPR, TPR, label="{} pixel-level ROCAUC: {}".format(category, pixel_rocauc))

        roc_score_save_directory = os.path.join(args.save_directory,
                                                "temp_{}_Layer_{}_Rd_{}".format(args.backbone_model, args.layer_used,
                                                                                args.dimensionality_reduction,
                                                                                category), category)
        with open(os.path.join(roc_score_save_directory, "rocauc_score.txt"), 'w') as f:
            f.write("image-level ROCAUC: {}\n".format(img_rocauc))
            f.write("pixel-level ROCAUC: {}".format(pixel_rocauc))

        # calculate optimal threshold for final results
        precision, recall, thresholds = precision_recall_curve(test_mask_list.flatten(), normalized_scores.flatten())
        f1_score = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(2 * precision * recall),
                             where=(precision + recall) != 0)
        threshold = thresholds[np.argmax(f1_score)]

        # final results for category
        os.makedirs(os.path.join(args.save_directory,
                                 "temp_{}_Layer_{}_Rd_{}".format(args.backbone_model, args.layer_used,
                                                                 args.dimensionality_reduction), category, "results"),
                    exist_ok=True)
        plot_fig(test_img_list, normalized_scores, test_mask_list, threshold, os.path.join(args.save_directory,
                                                                                           "temp_{}_Layer_{}_Rd_{}".format(
                                                                                               args.backbone_model,
                                                                                               args.layer_used,
                                                                                               args.dimensionality_reduction),
                                                                                           category, "results"),
                 category)

    # average result of model
    average_roc_score_save_directory = os.path.join(args.save_directory,
                                                    "temp_{}_Layer_{}_Rd_{}".format(args.backbone_model,
                                                                                    args.layer_used,
                                                                                    args.dimensionality_reduction,
                                                                                    category))
    with open(os.path.join(average_roc_score_save_directory, "rocauc_score.txt"), 'w') as f:
        f.write("Average image-level ROCAUC (all classes): {}\n".format(np.mean(total_img_rocauc)))
        f.write("Average pixel-level ROCAUC (all classes): {}\n".format(np.mean(total_pixel_rocauc)))

        f.write("Average image-level ROCAUC (all texture classes): {}\n".format(
            np.mean([total_img_rocauc[i] for i in [3, 4, 6, 10, 13]])))
        f.write("Average pixel-level ROCAUC (all texture classes): {}\n".format(
            np.mean([total_pixel_rocauc[i] for i in [3, 4, 6, 10, 13]])))

        f.write("Average image-level ROCAUC (all object classes): {}\n".format(
            np.mean([total_img_rocauc[i] for i in [0, 1, 2, 5, 7, 8, 9, 11, 12, 14]])))
        f.write("Average pixel-level ROCAUC (all object classes): {}\n".format(
            np.mean([total_pixel_rocauc[i] for i in [0, 1, 2, 5, 7, 8, 9, 11, 12, 14]])))

    print("Average image-level ROCAUC: {}".format(np.mean(total_img_rocauc)))
    fig_img_rocauc.title.set_text("Average image-level ROCAUC: {}".format(np.mean(total_img_rocauc)))
    fig_img_rocauc.legend(loc="lower right")

    print("Average pixel-level ROCAUC: {}".format(np.mean(total_pixel_rocauc)))
    fig_pixel_rocauc.title.set_text("Average pixel-level ROCAUC: {}".format(np.mean(total_pixel_rocauc)))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_directory, "temp_{}_Layer_{}_Rd_{}".format(args.backbone_model, args.layer_used,
                                                                                  args.dimensionality_reduction),
                             'roc_curve.png'), dpi=100)


def concat_embedding_vectors(x, y):
    batch_size, c_x, h_x, w_x = x.size()
    _, c_y, h_y, w_y = y.size()
    s = int(h_x / h_y)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(batch_size, c_x, -1, h_y, w_y)
    z = torch.zeros(batch_size, c_x + c_y, x.size(2), h_y, w_y)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(batch_size, -1, h_y * w_y)
    z = F.fold(z, kernel_size=s, output_size=(h_x, w_x), stride=s)
    return z


def get_reduced_vector_size_per_layers(reduced, layer_1_size, layer_2_size, layer_3_size):
    r = 1 / (layer_1_size + layer_2_size + layer_3_size)
    reduced_layer_1_size = int(reduced * r * layer_1_size)
    reduced_layer_2_size = int(reduced * r * layer_2_size)
    reduced_layer_3_size = reduced - reduced_layer_1_size - reduced_layer_2_size
    return reduced_layer_1_size, reduced_layer_2_size, reduced_layer_3_size


# https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()
