from .mvtec import *
from .AITEX import *
from .CustomDataset import *
from .BTAD import *

import requests
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.segmentation import mark_boundaries

BATCH_SIZE = 32             # batch size


def myPrint(string, filename):
    print(string)
    filename.write(string + '\n')

def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)


def telegram_bot_sendtext(bot_message):
    """
    Send a notice to a Telegram chat. To use, create a file "tg.ll" in the main folder with this form:
    {
    "token": "",    <-- bot token 
    "idchat": ""    <-- your chat id
    }
    """
    try:
        with open('./tg.ll') as f:
            data = json.load(f)
    except:
        print("ERROR: Can't send message on Telegram. Configure the \'./tg.ll\' file or set args.telegram=False.")
        return
    bot_token = data['token']
    bot_chatID = data['idchat']
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return str(response)


def precision(tp, fp):
    return tp/(tp + fp)

def sensitivity(tp, fn):    # True Positive Rate
    return tp/(tp + fn)

def FPR(fp, tn):            # False Positive Rate
    return fp/(fp + tn)

def F_score(precision, sensitivity, beta):
    return (1 + beta**2) * ((precision * sensitivity)/(beta**2 * precision + sensitivity))

def Binarization(mask, thres = 0., type = 0):
    if type == 0:
        mask = np.where(mask > thres, 1., 0.)
    elif type ==1:
        mask = np.where(mask > thres, mask, 0.)
    return mask


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
        img = bb.denormalization(img)
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