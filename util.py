import torch
from torchvision import transforms

from dataset import ImgDataset

def build_dataset(img_dir, mask_dir, isTrain=True):
    # train data augmentation
    transforms_train = transforms.Compose([    
        transforms.Resize(256),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        ])

    # test data augmentation
    transforms_test = transforms.Compose([  
        transforms.Resize(256),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        ])
    
    if(isTrain):
        imgDataset = ImgDataset(img_dir, mask_dir, transforms_train)
    else:
        imgDataset = ImgDataset(img_dir, mask_dir, transforms_test)

    return imgDataset 


def unnormalize(x):
    # currently normalization is not applied
    return x