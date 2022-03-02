import os
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import cv2
import shutil       # To copy the file
import sys
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision

import backbone as bb

customdataset_folder = './datasets/CustomDataset/'
customdataset_train_dir = customdataset_folder + 'trainset/'
customdataset_test_dir = customdataset_folder + 'testset/'
customdataset_mask_dir = customdataset_folder + 'Mask_images/'
customdataset_config_file = customdataset_folder + 'config'

CUSTOMDATASET_CLASS_NAMES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

PATCH_SIZE = 256            # patch size
STRIDE = PATCH_SIZE         # stride of patch
ANOMALY_THRESHOLD = 0

class CustomDataset(Dataset):
    def __init__(self, class_name='CustomDataset', resize=256, cropsize=224, is_train=True):
        self.is_train = is_train
        self.class_name = class_name
        self.resize = resize
        self.cropsize = cropsize
        self.transform = transforms.Compose([transforms.Resize(resize, Image.ANTIALIAS),
                                                transforms.CenterCrop(cropsize),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = transforms.Compose([transforms.Resize(resize, Image.NEAREST),
                                                    transforms.CenterCrop(cropsize),
                                                    transforms.ToTensor()])
        self.main_dir = customdataset_train_dir+self.class_name+'/' if self.is_train else customdataset_test_dir+self.class_name+'/'
        self.all_imgs = sorted(os.listdir(self.main_dir))
        self.mask_dir = customdataset_mask_dir
        if not self.is_train:
            self.all_mask = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)                                    # x in mvtec class
        mask_name = self.all_imgs[idx].replace('.png', '_mask.png')
        if os.path.isfile(self.mask_dir + mask_name):
            mask_loc = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_loc).convert('L')
            tensor_mask = self.transform_mask(mask)                             # mask in mvtec class
        else:
            tensor_mask = torch.zeros([1, PATCH_SIZE, PATCH_SIZE])              # y in mvtec class
        if int(torch.sum(tensor_mask)) > ANOMALY_THRESHOLD:
            defective = 1
        else:
            defective = 0
        return tensor_image, defective, tensor_mask

    def getName(self, idx, mask=False):
        if mask:
            return self.all_imgs[idx].replace('.png', '_mask.png')
        else:
            return self.all_imgs[idx]


def resizeCustomDataset(img):
    if img.shape[2] % PATCH_SIZE != 0:              # width
        patches_in_image = int(np.floor(img.shape[2] / PATCH_SIZE))
        new_width = img.shape[2] - (img.shape[2] - (patches_in_image * PATCH_SIZE))
    else:
        new_width = img.shape[2]
    if img.shape[1] % PATCH_SIZE != 0:              # height
        patches_in_image = int(np.floor(img.shape[1] / PATCH_SIZE))
        new_height = img.shape[1] - (img.shape[1] - (patches_in_image * PATCH_SIZE))
    else:
        new_height = img.shape[1]
    transform = transforms.CenterCrop((new_height, new_width))
    crop_img = transform(img)

    return crop_img, new_width, new_height


# --------------- Functions for patches ---------------

def DivideInPatches(img, size, stride):
    p = img.unfold(1, size, stride).unfold(2, size, stride)
    patches = p.contiguous().view(p.size(0), -1, size, size).permute(1,0,2,3)
    return patches


# --------------- Functions to create Custom Dataset ---------------

def DeleteFolder(path):
    shutil.rmtree(path)

def BinarizeMasks(Mask_path):
    thresh = 128
    maxval = 255

    all_imgs = sorted(os.listdir(Mask_path))
    for i in all_imgs:
        im_gray = np.array(Image.open(Mask_path+i).convert('L'))
        im_bin = (im_gray > thresh) * maxval
        Image.fromarray(np.uint8(im_bin)).save(Mask_path+i)

def RenameFolder(oldname, newname):
    os.rename(oldname, newname)

def CreateCustomDataset(log_file):
    try:
        BinarizeMasks(customdataset_mask_dir)
        train_folder_temp = customdataset_folder + 'trainset_temp/'
        test_folder_temp = customdataset_folder + 'testset_temp/'
        Mask_path_temp = customdataset_folder + 'Mask_images_temp/'
        RenameFolder(customdataset_train_dir, train_folder_temp)
        RenameFolder(customdataset_test_dir, test_folder_temp)
        RenameFolder(customdataset_mask_dir, Mask_path_temp)
        os.makedirs(customdataset_train_dir, exist_ok=True)
        os.makedirs(customdataset_test_dir, exist_ok=True)
        os.makedirs(customdataset_mask_dir, exist_ok=True)
        for Class in CUSTOMDATASET_CLASS_NAMES:
            os.makedirs(customdataset_train_dir+Class+'/', exist_ok=True)
            os.makedirs(customdataset_test_dir+Class+'/', exist_ok=True)
            
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        for Class in CUSTOMDATASET_CLASS_NAMES:
            train_temp = train_folder_temp+Class+'/'
            test_temp = test_folder_temp+Class+'/'
            all_train_imgs = sorted(os.listdir(train_temp))
            for img in all_train_imgs:
                img_loc = os.path.join(train_temp, img)
                image = Image.open(img_loc).convert('RGB')
                tensor_image = transform(image)
                tensor_image, _, _ = resizeCustomDataset(tensor_image)
                train_patches = DivideInPatches(tensor_image, PATCH_SIZE, STRIDE)
                for idx, patch in enumerate(train_patches):
                    name = img.replace('.png', '_'+str(idx)+'.png')
                    name = os.path.join(customdataset_train_dir+Class+'/', name)
                    torchvision.utils.save_image(patch, name)
            all_test_imgs = sorted(os.listdir(test_temp))
            for img in all_test_imgs:
                img_loc = os.path.join(test_temp, img)
                image = Image.open(img_loc).convert('RGB')
                tensor_image = transform(image)
                tensor_image, new_width, new_height = resizeCustomDataset(tensor_image)
                test_patches = DivideInPatches(tensor_image, PATCH_SIZE, STRIDE)
                for idx, patch in enumerate(test_patches):
                    name = img.replace('.png', '_'+str(idx)+'.png')
                    name = os.path.join(customdataset_test_dir+Class+'/', name)
                    torchvision.utils.save_image(patch, name)
                mask_name = img.replace('.png', '_mask.png')
                if os.path.isfile(Mask_path_temp + mask_name):
                    mask_loc = os.path.join(Mask_path_temp, mask_name)
                    mask = Image.open(mask_loc).convert('L')
                    tensor_mask = transform(mask)
                else:
                    tensor_mask = torch.zeros([1, new_height, new_width])
                transform_mask = transforms.CenterCrop((new_height, new_width))
                tensor_mask = transform_mask(tensor_mask)
                test_masks = DivideInPatches(tensor_mask, PATCH_SIZE, STRIDE)
                for idx, patch in enumerate(test_masks):
                    name = mask_name.replace('_mask.png', '_'+str(idx)+'_mask.png')
                    name = os.path.join(customdataset_mask_dir, name)
                    torchvision.utils.save_image(patch, name)

        DeleteFolder(train_folder_temp)
        DeleteFolder(test_folder_temp)
        DeleteFolder(Mask_path_temp)
        f = open(customdataset_config_file, "a")
        f.write("This file indicates that the dataset is ready to be used. Don't delete it!")
        f.close()
    except:
        bb.myPrint("Error in CreateCustomDataset function!", log_file)
        DeleteFolder(customdataset_folder)
        sys.exit(-1)


def prepareCustomDataset(log_file):
    if os.path.isdir(customdataset_folder):
        if (os.path.exists(customdataset_config_file)):
            return
        else:
            bb.myPrint("Preparing the Custom Dataset...", log_file)
            CreateCustomDataset(log_file)
    else:
        bb.myPrint('ERROR: Custom Dataset not found!', log_file)
        sys.exit(-1)
