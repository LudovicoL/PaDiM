import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import shutil
import backbone as bb

URL = 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    def __init__(self, log_file, dataset_path='./datasets/MVTecAD', class_name='carpet', is_train=True,
                 resize=256, cropsize=224):
        assert class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.log_file = log_file
        # download dataset if not exist
        if not os.path.isdir(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            self.download()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            
            x.extend(img_fpath_list)
            
            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
        
        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)

    def download(self):
        try:
            """Download dataset if not exist"""
            tar_file_path = self.dataset_path + '/mvtec_anomaly_detection.tar.xz'
            if not os.path.exists(tar_file_path):
                download_url(URL, tar_file_path)
            bb.myPrint('unzip dataset: %s' % tar_file_path, self.log_file)
            tar = tarfile.open(tar_file_path, 'r:xz')
            tar.extractall(self.dataset_path)
            tar.close()
            os.remove(tar_file_path)
        except Exception as e:
            bb.myPrint(e, self.log_file)
            bb.myPrint("Can't download MVTecAD dataset. Retry later.", self.log_file)
            shutil.rmtree(self.dataset_path)
            sys.exit(-1)
        return


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
