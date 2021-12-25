from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import cv2
import random
from torchvision.utils import make_grid
from torchvision.utils import save_image

import util

"""
Wrapper class for Places2 Dateset
"""
class ImgDataset(Dataset):
    """
    img_dir: directory for original train/val images
    mask_dir: directory for train/val masks
    """
    def __init__(self, img_dir, mask_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.img_names = {i:name for i,name in enumerate(os.listdir(img_dir))}
        self.mask_names = {i:name for i,name in enumerate(os.listdir(mask_dir))}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir,
                     self.mask_names[random.randint(0, len(self.mask_names)-1)])

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path) 
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if(self.transform):
            img = self.transform(img)
            mask = self.transform(mask)

        # masked image, mask, groud truth image
        return img*mask, mask, img


#### For testing
if __name__ == '__main__':
	imgDataset = util.build_dataset('./dataset/train','./dataset/train_mask')
	print(len(imgDataset))
	img, mask, gt = zip(*[imgDataset[i] for i in range(5)])
	img = torch.stack(img, dim=0) # --> i x 3 x 256 x 256
	mask = torch.stack(mask, dim=0)
	gt = torch.stack(gt, dim=0)

	grid = make_grid(torch.cat((util.unnormalize(img), mask, util.unnormalize(gt)), dim=0), nrow=5)
	save_image(grid, "./result/input_show.jpg")