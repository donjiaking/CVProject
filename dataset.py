from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import cv2
import random
from PIL import Image
from torchvision.utils import make_grid
from torchvision.utils import save_image

import util


def build_dataset(img_dir, mask_dir, isTrain=True):
    ## train data augmentation
    transform_train_img = transforms.Compose([    
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip()
        ])
    transform_train_msk = transforms.Compose([    
        transforms.ToTensor(),
        transforms.Resize(256),
        ])

    ## test data augmentation
    transform_test_img = transforms.Compose([  
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    transform_test_msk = transforms.Compose([  
        transforms.ToTensor(),
        transforms.Resize(256),
        ])
    
    if(isTrain):
        imgDataset = ImgDataset(img_dir, mask_dir, transform_train_img, transform_train_msk)
    else:
        imgDataset = ImgDataset(img_dir, mask_dir, transform_test_img, transform_test_msk)

    return imgDataset 


"""
Wrapper class for Places2 Dateset
"""
class ImgDataset(Dataset):
    """
    img_dir: directory for original train/val images
    mask_dir: directory for masks
    """
    def __init__(self, img_dir, mask_dir, transform_img, transform_msk):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_msk = transform_msk

        self.img_names = {i:name for i,name in enumerate(os.listdir(img_dir))}
        self.mask_names = {i:name for i,name in enumerate(os.listdir(mask_dir))}

    def __len__(self):
        # return len(self.img_names)
        return 10000

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir,
                     self.mask_names[random.randint(0, len(self.mask_names)-1)])

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path) 
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if(self.transform_img):
            img = self.transform_img(img)
        if(self.transform_msk):
            mask = self.transform_msk(mask)

        # masked image, mask, groud truth image
        return img*mask, mask, img


#### For testing
if __name__ == '__main__':
    imgDataset = build_dataset('./dataset/train','./dataset/masks')
    print(len(imgDataset))
    nrow = 5
    img, mask, gt = zip(*[imgDataset[i] for i in range(nrow)])
    img = torch.stack(img, dim=0) # shape: nrow x 3 x 256 x 256
    mask = torch.stack(mask, dim=0)
    gt = torch.stack(gt, dim=0)

    grid = make_grid(torch.cat((util.unnormalize(gt)*mask, mask, util.unnormalize(gt)), dim=0), nrow=nrow)
    save_image(grid, "result/input_show.jpg")