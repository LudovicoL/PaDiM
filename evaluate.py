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
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import sys
import time
import backbone as bb


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# log directory
log_folders = sorted(list(bb.folders_in('./outputs/')))
if not os.path.isdir('./outputs/') or len(log_folders) == 0:
    print('Run \'train_network.py\' first!')
    sys.exit(-1)
log_dir = log_folders[-1] + '/'


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("-d", "--dataset", default="aitex", help="Choose the dataset: \"aitex\", \"mvtec\", \"btad\".")
    parser.add_argument("-t", '--telegram', default=False, action="store_true", help="Send notification on Telegram.")
    parser.add_argument("-a", '--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument("-b", '--batch_size', default=bb.BATCH_SIZE, help="Set batch size.")
    return parser.parse_args()

def main():
    start_time = time.time()
    test_times = []
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
        number_of_defects, _ = bb.countAitexAnomalies()
        bb.myPrint("There are " + str(number_of_defects) + " images with defects.", log_file)
        CLASS_NAMES = bb.AITEX_CLASS_NAMES
    elif args.dataset == "btad":
        CLASS_NAMES = bb.BTAD_CLASS_NAMES
    elif args.dataset == "custom":
        CLASS_NAMES = bb.CUSTOMDATASET_CLASS_NAMES
    else:
        bb.myPrint("Error! Choose a valid dataset.", log_file)
        sys.exit(-1)
    
    for class_name in CLASS_NAMES:
        if args.dataset == "mvtec":
            test_dataset = bb.MVTecDataset(class_name=class_name, is_train=False, log_file=log_file)
        elif args.dataset == "aitex":
            test_dataset = bb.AitexDataSet(is_train=False, class_name=class_name)
        elif args.dataset == "btad":
            test_dataset = bb.BtadDataset(is_train=False, class_name=class_name)
        elif args.dataset == "custom":
            test_dataset = bb.CustomDataset(class_name=class_name, is_train=False)
        
        bb.myPrint("There are "+str(len(test_dataset))+" test images for "+str(class_name)+" class.", log_file)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(log_dir, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        bb.myPrint('load train set feature from: %s' % train_feature_filepath, log_file)
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)

        test_time = time.time()    # Measure the test time

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
            embedding_vectors = bb.embedding_concat(embedding_vectors, test_outputs[layer_name])
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
        test_times.append(time.time() - test_time)
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
        tp, tn, fp, fn = bb.plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)
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

    bb.myPrint("Average test time: %s seconds." % np.mean(test_times), log_file)
    bb.myPrint("---Execution time: %s seconds ---\n" % (time.time() - start_time), log_file)
    log_file.close()
    if args.telegram: bb.telegram_bot_sendtext("*PaDiM*:\nAverage ROCAUC: _"+str(total_roc_auc) + "_\nAverage pixel ROCUAC: _"+str(total_pixel_roc_auc)+"_")





if __name__ == '__main__':
    main()

