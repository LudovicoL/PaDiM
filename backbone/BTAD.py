import os
from PIL import Image
import shutil
import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import backbone as bb

btad_folder = './datasets/BTAD/'
btad_config_file = btad_folder + 'config'

BTAD_CLASS_NAMES = ['01', '02', '03']

ANOMALY_THRESHOLD = 0

class BtadDataset(Dataset):
    def __init__(self, class_name, resize=256, cropsize=224, is_train=True):
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
        self.main_dir = btad_folder+self.class_name+'/train/' if self.is_train else btad_folder+self.class_name+'/test/'
        self.all_imgs = sorted(os.listdir(self.main_dir))
        self.mask_dir = btad_folder+self.class_name+'/ground_truth/'
        if not self.is_train:
            self.all_mask = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)                                    # x in mvtec class
        mask_name = self.all_imgs[idx]
        if os.path.isfile(self.mask_dir + mask_name):
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



def CreateBtadDataset(log_file):
    try:
        for Class in BTAD_CLASS_NAMES:
            train_folder = btad_folder + Class + '/train/'
            train_ok_folder = train_folder + 'ok/'
            all_train_imgs = sorted(os.listdir(train_ok_folder))
            for img in all_train_imgs:
                if img.split('.')[1] == 'bmp':
                    img_ = img.replace('.bmp', '.png')
                    os.rename(train_ok_folder + img, train_ok_folder + img_)
                    img = img_
                shutil.move(train_ok_folder + img, train_folder + img)
            shutil.rmtree(train_ok_folder)
            test_folder = btad_folder + Class + '/test/'
            test_ok_folder = test_folder + 'ok/'
            test_ko_folder = test_folder + 'ko/'
            all_test_ko_imgs = sorted(os.listdir(test_ko_folder))
            for img in all_test_ko_imgs:
                if img.split('.')[1] == 'bmp':
                    img_ = img.replace('.bmp', '.png')
                    os.rename(test_ko_folder + img, test_ko_folder + img_)
                    img = img_
                shutil.move(test_ko_folder + img, test_folder + img)
            shutil.rmtree(test_ko_folder)
            last_test_img = all_test_ko_imgs[-1]
            last_number = int(last_test_img.split('.')[0])
            for img in sorted(os.listdir(test_ok_folder)):
                last_number += 1
                if last_number<100:
                    new_name = '00'+str(last_number)+'.png'
                elif last_number<1000:
                    new_name = '0'+str(last_number)+'.png'
                shutil.move(test_ok_folder + img, test_folder + new_name)
            shutil.rmtree(test_ok_folder)
            masks_folder = btad_folder + Class + '/ground_truth/'
            mask_ko_folder = masks_folder + 'ko/'
            all_masks_imgs = sorted(os.listdir(mask_ko_folder))
            for img in all_masks_imgs:
                if img.split('.')[1] == 'bmp':
                    img_ = img.replace('.bmp', '.png')
                    os.rename(mask_ko_folder + img, mask_ko_folder + img_)
                    img = img_
                shutil.move(mask_ko_folder + img, masks_folder + img)
            shutil.rmtree(mask_ko_folder)
        f = open(btad_config_file, "a")
        f.write("This file indicates that the dataset is ready to be used. Don't delete it!")
        f.close()
    except Exception as e:
        bb.myPrint(e, log_file)
        bb.myPrint("Error in CreateBtadDataset function!", log_file)
        shutil.rmtree(btad_folder)
        sys.exit(-1)


def prepareBtad(log_file):
    if os.path.isdir(btad_folder) or os.path.isdir('./datasets/BTech_Dataset_transformed'):
        if os.path.isdir('./datasets/BTech_Dataset_transformed'):
            os.rename('./datasets/BTech_Dataset_transformed', btad_folder)
        if (os.path.exists(btad_config_file)):
            return
        else:
            bb.myPrint("Preparing the BTAD Dataset...", log_file)
            CreateBtadDataset(log_file)
    else:
        bb.myPrint('ERROR: BTAD not found!', log_file)
        sys.exit(-1)