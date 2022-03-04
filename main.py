import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import sys
from datetime import datetime
import time
import backbone as bb

BATCH_SIZE = 32             # batch size

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
os.makedirs('./outputs/', exist_ok=True)
os.makedirs('./datasets/', exist_ok=True)

date = datetime.now()
date = date.strftime("%Y-%m-%d_%H-%M-%S")
log_dir = './outputs/' + date + '/'
os.mkdir(log_dir)


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("-d", "--dataset", default="aitex", help="Choose the dataset: \"aitex\", \"mvtec\", \"btad\".")
    parser.add_argument("-t", '--telegram', default=False, action="store_true", help="Send notification on Telegram.")
    parser.add_argument("-a", '--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument("-r", '--resize', default=False, action="store_true", help="Resize AITEX dataset.")
    return parser.parse_args()

def main():
    start_time = time.time()
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    args = parse_args()
    log_file = open(log_dir + "log.txt", "a")
    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))
    
    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(log_dir, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    bb.myPrint("Dataset used: "+args.dataset, log_file)
    if args.dataset == "mvtec":
        CLASS_NAMES = bb.MVTEC_CLASS_NAMES
    elif args.dataset == "aitex":
        bb.myPrint("Resize: " + str(args.resize), log_file)
        bb.prepareAitex(args.resize, log_file)
        number_of_defects, _ = bb.countAitexAnomalies()
        bb.myPrint("There are " + str(number_of_defects) + " images with defects.", log_file)
        CLASS_NAMES = bb.AITEX_CLASS_NAMES
    elif args.dataset == "btad":
        bb.prepareBtad(log_file)
        CLASS_NAMES = bb.BTAD_CLASS_NAMES
    elif args.dataset == "custom":
        bb.prepareCustomDataset(log_file)
        CLASS_NAMES = bb.CUSTOMDATASET_CLASS_NAMES
    else:
        bb.myPrint("Error! Choose a valid dataset.", log_file)
        sys.exit(-1)
    
    for class_name in CLASS_NAMES:

        if args.dataset == "mvtec":
            train_dataset = bb.MVTecDataset(class_name=class_name, is_train=True, log_file=log_file)
            test_dataset = bb.MVTecDataset(class_name=class_name, is_train=False, log_file=log_file)
        elif args.dataset == "aitex":
            train_dataset = bb.AitexDataSet(is_train=True, class_name=class_name)
            test_dataset = bb.AitexDataSet(is_train=False, class_name=class_name)
        elif args.dataset == "btad":
            train_dataset = bb.BtadDataset(is_train=True, class_name=class_name)
            test_dataset = bb.BtadDataset(is_train=False, class_name=class_name)
        elif args.dataset == "custom":
            train_dataset = bb.CustomDataset(class_name=class_name, is_train=True)
            test_dataset = bb.CustomDataset(class_name=class_name, is_train=False)
        
        bb.myPrint("There are "+str(len(train_dataset))+" train images for "+str(class_name)+" class.", log_file)
        bb.myPrint("There are "+str(len(test_dataset))+" test images for "+str(class_name)+" class.", log_file)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(log_dir, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            bb.myPrint('load train set feature from: %s' % train_feature_filepath, log_file)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        bb.myPrint('test | %s |' % class_name, log_file)
        bb.myPrint('image ROCAUC: %.3f' % (img_roc_auc), log_file)
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        bb.myPrint('pixel ROCAUC: %.3f' % (per_pixel_rocauc), log_file)

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = log_dir + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        tp, tn, fp, fn = plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)
        true_positive += tp
        true_negative += tn
        false_positive += fp
        false_negative += fn

    total_roc_auc = np.mean(total_roc_auc)
    total_pixel_roc_auc = np.mean(total_pixel_roc_auc)

    bb.myPrint('Average ROCAUC: %.3f' % total_roc_auc, log_file)
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % total_roc_auc)
    fig_img_rocauc.legend(loc="lower right")

    bb.myPrint('Average pixel ROCUAC: %.3f' % total_pixel_roc_auc, log_file)
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % total_pixel_roc_auc)
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, 'roc_curve.png'), dpi=100)

    bb.myPrint('True positive: ' + str(true_positive), log_file)
    bb.myPrint('True negative: ' + str(true_negative), log_file)
    bb.myPrint('False positive: ' + str(false_positive), log_file)
    bb.myPrint('False negative: ' + str(false_negative), log_file)

    precision = bb.precision(true_positive, false_positive)
    bb.myPrint('Precision: ' + str(precision), log_file)
    sensitivity = bb.sensitivity(true_positive, false_negative)
    bb.myPrint('Sensitivity: ' + str(sensitivity), log_file)
    bb.myPrint('False Positive Rate: ' + str(bb.FPR(false_positive, true_negative)), log_file)
    bb.myPrint('F1-Score: ' + str(bb.F_score(precision, sensitivity, beta=1)), log_file)
    bb.myPrint('F2-Score: ' + str(bb.F_score(precision, sensitivity, beta=2)), log_file)

    bb.myPrint("---Execution time: %s seconds ---\n" % (time.time() - start_time), log_file)
    log_file.close()
    if args.telegram: bb.telegram_bot_sendtext("*PaDiM*:\nAverage ROCAUC: _"+str(total_roc_auc) + "_\nAverage pixel ROCUAC: _"+str(total_pixel_roc_auc)+"_")



def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        
        if np.sum(gt) > 0 and np.sum(mask) > 0:
            tp +=1
        elif np.sum(gt) == 0 and np.sum(mask) == 0:
            tn +=1
        elif np.sum(gt) == 0 and np.sum(mask) > 0:
            fp += 1
        elif np.sum(gt) > 0 and np.sum(mask) == 0:
            fn += 1
        
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
    return tp, tn, fp, fn     


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()

