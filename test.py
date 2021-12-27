from numpy.core.fromnumeric import size
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import make_grid
from torchvision.utils import save_image
import time
import os
import argparse

import util
from net import PConvUNet

##added 2021/12/27
device = torch.device('cuda')

"""
Evaluate the model given a dataset
"""
def evaluate(model, testDataset, args):
    test_loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

    print("Testing Started")  
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            print("Process: Batch ", i)
            # get the input
            img, mask, gt = data
            # get the output
            output = model(img, mask)
            # TODO: Erro Function?

"""
Show masked image, mask, output, groud truth image in one grid
"""
def show_visual_result(model, dataset, output_dir, n=6):
    img, mask, gt = zip(*[dataset[i] for i in range(n)])

    # for i in range(n):
    #     img[i] = img[i].to(device)
    #     mask[i] = mask[i].to(device)
    #     gt[i] = gt[i].to(device)
    img = torch.stack(img, dim=0) # --> n x 3 x 256 x 256
    mask = torch.stack(mask, dim=0)
    gt = torch.stack(gt, dim=0)
    img = img.to(device)
    mask = mask.to(device)
    gt = gt.to(device)
    with torch.no_grad():
        output = model(img, mask)

    grid = make_grid(torch.cat((util.unnormalize(gt)*mask, mask, util.unnormalize(output), util.unnormalize(gt)), dim=0), nrow=n)

    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    save_image(grid, os.path.join(output_dir, "output_show.jpg"))

    print("Result successfully saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)#when in gtx960, the batch_size will be set to 4
    parser.add_argument('--img_dir', type=str, default="./dataset/val")
    parser.add_argument('--mask_dir', type=str, default="./dataset/masks")
    parser.add_argument('--model_dir', type=str, default="./model")
    parser.add_argument('--out_dir', type=str, default="./result")
    args = parser.parse_args()

    model = PConvUNet().to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth")))
    testDataset = util.build_dataset(args.img_dir, args.mask_dir, isTrain=False)

    # evaluate(model, testDataset, args)
    show_visual_result(model, testDataset, args.out_dir)
