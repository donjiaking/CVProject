import torch
from torchvision import transforms

from dataset import ImgDataset

##added 2021/12/27
device = torch.device('cuda')

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


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor([0.229, 0.224, 0.225]).to(device) + \
            torch.Tensor([0.485, 0.456, 0.406]).to(device)
    x = x.transpose(1, 3)

    return x