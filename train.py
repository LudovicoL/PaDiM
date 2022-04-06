import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import sys
from datetime import datetime
import time
import backbone as bb



# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# log directory
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
    parser.add_argument("-b", '--batch_size', default=bb.BATCH_SIZE, help="Set batch size.")
    return parser.parse_args()

def main():
    start_time = time.time()
    train_times = []
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

    bb.myPrint("Dataset used: "+args.dataset, log_file)
    if args.dataset == "mvtec":
        CLASS_NAMES = bb.MVTEC_CLASS_NAMES
    elif args.dataset == "aitex":
        bb.myPrint("Resize: " + str(args.resize), log_file)
        bb.prepareAitex(args.resize, log_file)
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
        elif args.dataset == "aitex":
            train_dataset = bb.AitexDataSet(is_train=True, class_name=class_name)
        elif args.dataset == "btad":
            train_dataset = bb.BtadDataset(is_train=True, class_name=class_name)
        elif args.dataset == "custom":
            train_dataset = bb.CustomDataset(class_name=class_name, is_train=True)
        
        bb.myPrint("There are "+str(len(train_dataset))+" train images for "+str(class_name)+" class.", log_file)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        train_time = time.time()    # Measure the train time

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
                embedding_vectors = bb.embedding_concat(embedding_vectors, train_outputs[layer_name])
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

        train_times.append(time.time() - train_time)
        

    bb.myPrint("Average train time: %s seconds." % np.mean(train_times), log_file)
    bb.myPrint("---Execution time: %s seconds ---\n" % (time.time() - start_time), log_file)
    log_file.close()
    if args.telegram: bb.telegram_bot_sendtext("*PaDiM*: training finished.")


if __name__ == '__main__':
    main()

