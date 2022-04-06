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
from tqdm import tqdm
import urllib.request
import py7zr

import backbone as bb


aitex_folder = './datasets/AITEX/'
aitex_train_dir = aitex_folder + 'trainset/'
aitex_test_dir = aitex_folder + 'testset/'
aitex_mask_dir = aitex_folder + 'Mask_images/'
aitex_config_file = aitex_folder + 'config'

Defect_path = aitex_folder + 'Defect_images/'
NODefect_path = aitex_folder + 'NODefect_images/'
CUT_PATCHES = 1

AITEX_CLASS_NAMES = ['00', '01', '02', '03', '04', '05', '06']

PATCH_SIZE = 256            # patch size
STRIDE = PATCH_SIZE         # stride of patch
ANOMALY_THRESHOLD = 0       # threshold to consider a patch as anomalous


class AitexDataSet(Dataset):
    def __init__(self, class_name='03', resize=256, cropsize=224, is_train=True):
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
        self.main_dir = aitex_train_dir+self.class_name+'/' if self.is_train else aitex_test_dir+self.class_name+'/'
        self.all_imgs = sorted(os.listdir(self.main_dir))
        self.mask_dir = aitex_mask_dir
        if not self.is_train:
            self.all_mask = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)                                    # x in mvtec class
        mask_name = self.all_imgs[idx].replace('.png', '_mask.png')
        if os.path.isfile(self.mask_dir + '/' + mask_name):
            mask_loc = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_loc).convert('L')
            tensor_mask = self.transform_mask(mask)                             # mask in mvtec class
        else:
            tensor_mask = torch.zeros([1, self.cropsize, self.cropsize])
        if int(torch.sum(tensor_mask)) > ANOMALY_THRESHOLD:                     # y in mvtec class
            defective = 1
        else:
            defective = 0
        return tensor_image, defective, tensor_mask

    def getName(self, idx, mask=False):
        if mask:
            return self.all_imgs[idx].replace('.png', '_mask.png')
        else:
            return self.all_imgs[idx]
    


def resizeAitex(dataset, original_width=4096, original_height=256):
    img_ = dataset.squeeze(0).numpy()
    img_ = cv2.normalize(img_, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img_ = img_.astype(np.uint8)
            
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_,(3,3), sigmaX=0, sigmaY=0) 

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    vector = np.zeros(original_width)
    for i in range(original_width):
        for j in range(original_height):
            vector[i] += edges[j][i]
    derivative = np.gradient(vector)
    max = np.argmax(derivative)
    cut = (int(max/PATCH_SIZE) + CUT_PATCHES) * PATCH_SIZE

    crop_img = transforms.functional.crop(dataset, top=0, left=cut, height=original_height, width=(original_width-cut))
    new_widths = crop_img.shape[2]
    new_heights = crop_img.shape[1] 

    return crop_img, new_widths, new_heights



# --------------- Functions for patches ---------------

def DivideInPatches(img, size, stride):
    patches = img.unfold(1, size, stride).unfold(2, size, stride)
    patches = patches.contiguous().view(patches.size(0), -1, size, size).permute(1,0,2,3)
    return patches

def countAitexAnomalies():
    masks = sorted(os.listdir(aitex_mask_dir))
    number_of_defects = 0
    defective = []
    transform_mask = transforms.ToTensor()
    for mask_name in masks:
        mask_loc = os.path.join(aitex_mask_dir, mask_name)
        mask = Image.open(mask_loc).convert('L')
        tensor_mask = transform_mask(mask)
        if int(torch.sum(tensor_mask)) > ANOMALY_THRESHOLD:
            number_of_defects += 1
            defective.append(True)
        else:
            defective.append(False)
    return number_of_defects, defective


# --------------- Functions to create Aitex Dataset ---------------

def Reformat_Image(ImageFilePath, new_width, new_height, color, offset):

    image = Image.open(ImageFilePath, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if color == 'white':
        color = (255, 255, 255, 255)
    elif color == 'black':
        color = (0, 0, 0, 255)

    if offset == 'center':
        offset = (int(round(((new_width - width) / 2), 0)), int(round(((new_height - height) / 2), 0)))
    elif offset == 'right':
        offset = (0, 0)
    elif offset == 'left':
        offset = ((new_width - width), (new_height - height))

    background = Image.new('RGBA', (new_width, new_height), color)

    background.paste(image, offset)
    background.save(ImageFilePath)

def DeleteFolder(path):
    shutil.rmtree(path)

def MergeMasks(name):
    mask1 = Image.open(name+'_mask1.png').convert('L')
    mask2 = Image.open(name+'_mask2.png').convert('L')
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    mask = np.add(mask1, mask2)
    mask = Image.fromarray(mask)
    mask.save(name+'_mask.png',"png")
    os.remove(name+'_mask1.png')
    os.remove(name+'_mask2.png')

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

def FlipImage(filename):
    image = Image.open(filename)
    image = np.fliplr(image)
    Image.fromarray(np.uint8(image)).save(filename)

def CreateAitexDataset(resize, log_file):
    try:
        bb.myPrint("Preparing the AITEX dataset...", log_file)
        NODefect_subdirectories = {
            '2311694-2040n7u': '00',
            '2608691-202020u': '01',
            '2306894-210033u': '02',
            '2311694-1930c7u': '03',
            '2311517-195063u': '04',
            '2306881-210020u': '05',
            '2311980-185026u': '06'
            }
            
        os.makedirs(aitex_train_dir, exist_ok=True)
        os.makedirs(aitex_test_dir, exist_ok=True)

        for i in range(len(NODefect_subdirectories)):
            RenameFolder(NODefect_path+list(NODefect_subdirectories.keys())[i], NODefect_path+list(NODefect_subdirectories.values())[i])
            os.makedirs(aitex_train_dir+list(NODefect_subdirectories.values())[i], exist_ok=True)
            os.makedirs(aitex_test_dir+list(NODefect_subdirectories.values())[i], exist_ok=True)
            
        MergeMasks(aitex_mask_dir+'0044_019_04')   # Merge and delete 0044_019_04.png masks
        MergeMasks(aitex_mask_dir+'0097_030_03')   # Merge and delete 0097_030_03.png masks
        BinarizeMasks(aitex_mask_dir)
        Reformat_Image(Defect_path + '0094_027_05.png', 4096, 256, 'white', 'right')
        Reformat_Image(aitex_mask_dir + '0094_027_05_mask.png', 4096, 256, 'black', 'right')
        os.remove(Defect_path + '0100_025_08.png')
        FlipImage(Defect_path + '0094_027_05.png')
        FlipImage(aitex_mask_dir + '0094_027_05_mask.png')

        defect_images = os.listdir(Defect_path)
        nodefect_images = []

        for i in range(len(NODefect_subdirectories)):
            for j in os.listdir(NODefect_path + list(NODefect_subdirectories.values())[i]):
                nodefect_images.append(list(NODefect_subdirectories.keys())[i] + '/' + j)
            
        for i in range(len(NODefect_subdirectories)):
            new_folder = Defect_path+list(NODefect_subdirectories.values())[i] + '/'
            os.makedirs(new_folder, exist_ok=True)
            for img in defect_images:
                if list(NODefect_subdirectories.values())[i]+'.png' in img:
                    shutil.move(Defect_path + img, new_folder + img)


        Mask_path_temp = aitex_folder + '/Mask_images_temp/'
        RenameFolder(aitex_mask_dir, Mask_path_temp)
        os.makedirs(aitex_mask_dir, exist_ok=True)

        for i in range(len(NODefect_subdirectories)):
            last_image = os.listdir(NODefect_path+list(NODefect_subdirectories.values())[i] + '/')[-1]
            new_folder = Defect_path+list(NODefect_subdirectories.values())[i] + '/' + last_image
            old_folder = NODefect_path+list(NODefect_subdirectories.values())[i] + '/' + last_image
            shutil.move(old_folder, new_folder)
            
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        for i in range(len(NODefect_subdirectories)):
            train_folder_temp = NODefect_path+list(NODefect_subdirectories.values())[i] + '/'
            all_train_imgs = sorted(os.listdir(train_folder_temp))
            for img in all_train_imgs:
                img_loc = os.path.join(train_folder_temp, img)
                image = Image.open(img_loc).convert('L')
                tensor_image = transform(image)
                if resize:
                    tensor_image, _, _ = resizeAitex(tensor_image)
                train_patches = DivideInPatches(tensor_image, PATCH_SIZE, STRIDE)
                for idx, patch in enumerate(train_patches):
                    name = img.replace('.png', '_'+str(idx)+'.png')
                    name = os.path.join(aitex_train_dir+list(NODefect_subdirectories.values())[i] + '/', name)
                    torchvision.utils.save_image(patch, name)
            test_folder_temp = Defect_path+list(NODefect_subdirectories.values())[i] + '/'
            all_test_imgs = sorted(os.listdir(test_folder_temp))
            for img in all_test_imgs:
                img_loc = os.path.join(test_folder_temp, img)
                image = Image.open(img_loc).convert('L')
                tensor_image = transform(image)
                if resize:
                    tensor_image, new_widths, _ = resizeAitex(tensor_image)
                test_patches = DivideInPatches(tensor_image, PATCH_SIZE, STRIDE)
                for idx, patch in enumerate(test_patches):
                    name = img.replace('.png', '_'+str(idx)+'.png')
                    name = os.path.join(aitex_test_dir+list(NODefect_subdirectories.values())[i] + '/', name)
                    torchvision.utils.save_image(patch, name)
                mask_name = img.replace('.png', '_mask.png')
                if os.path.isfile(Mask_path_temp + mask_name):
                    mask_loc = os.path.join(Mask_path_temp, mask_name)
                    mask = Image.open(mask_loc).convert('L')
                    tensor_mask = transform(mask)
                else:
                    tensor_mask = torch.zeros([1, 256, 4096])
                if resize:
                    tensor_mask = transforms.functional.crop(tensor_mask, top=0, left=(4096-new_widths), height=256, width=new_widths)
                test_masks = DivideInPatches(tensor_mask, PATCH_SIZE, STRIDE)
                for idx, patch in enumerate(test_masks):
                    name = mask_name.replace('_mask.png', '_'+str(idx)+'_mask.png')
                    name = os.path.join(aitex_mask_dir, name)
                    torchvision.utils.save_image(patch, name)

        DeleteFolder(Defect_path)
        DeleteFolder(NODefect_path)
        DeleteFolder(Mask_path_temp)
        f = open(aitex_config_file, "a")
        f.write(str(resize))
        f.close()
    except Exception as e:
        bb.myPrint(e, log_file)
        bb.myPrint("Error in CreateAitexDataset function!", log_file)
        DeleteFolder(aitex_folder)
        sys.exit(-1)

def prepareAitex(resize, log_file):
    if os.path.isdir(aitex_folder):
        if (os.path.isdir(aitex_train_dir) and os.path.isdir(aitex_test_dir) and os.path.isdir(aitex_mask_dir)):
            f = open(aitex_config_file, "r")
            resize_ = f.readline()
            f.close()
            resize_ = True if resize_ == "True" else False
            if resize == resize_:
                return
            else:
                DeleteFolder(aitex_folder)
                download(log_file)
                CreateAitexDataset(resize, log_file)
        else:
            DeleteFolder(aitex_folder)
            download(log_file)
            CreateAitexDataset(resize, log_file)
    else:
        download(log_file)
        CreateAitexDataset(resize, log_file)


# --------------- Functions to download Aitex Dataset ---------------

URL = 'https://www.aitex.es/wp-content/uploads/2019/07/'

ARCHIVES = [
    'Defect_images.7z',
    'NODefect_images.7z',
    'Mask_images.7z'
]

def download(log_file):
    bb.myPrint("Download AITEX dataset...", log_file)
    os.makedirs(aitex_folder, exist_ok=True)
    try:
        for idx in range(len(ARCHIVES)):
            if not os.path.isfile(aitex_folder+ARCHIVES[idx]):
                download_url(URL+ARCHIVES[idx], aitex_folder+ARCHIVES[idx])
            with py7zr.SevenZipFile(aitex_folder+ARCHIVES[idx], mode='r') as z:
                z.extractall(path=aitex_folder)
            os.remove(aitex_folder+ARCHIVES[idx])
        return
    except Exception as e:
        bb.myPrint(str(e), log_file)
        bb.myPrint("Can't download AITEX dataset. Retry later.", log_file)
        sys.exit(-1) 


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)