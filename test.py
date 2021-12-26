from numpy.core.fromnumeric import size
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import make_grid
from torchvision.utils import save_image
import time
import os

import util
from net import PConvUNet

TEST_IMG = './dataset/val'
TEST_MSK = './dataset/masks'

BATCH_SIZE = 32

"""
Evaluate the model given a dataset
"""
def evaluate(model, testDataset):
    test_loader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)

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
show masked image, mask, output, groud truth image in one grid
"""
def show_visual_result(model, dataset, output_dir, n=6):
    img, mask, gt = zip(*[dataset[i] for i in range(n)])
    img = torch.stack(img, dim=0) # --> n x 3 x 256 x 256
    mask = torch.stack(mask, dim=0)
    with torch.no_grad():
        output = model(img, mask)

    grid = make_grid(torch.cat((img, mask, output, gt), dim=0), nrow=n)

    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    save_image(grid, os.path.join(output_dir, "output_show.jpg"))

    print("Result successfully saved")


if __name__ == "__main__":
    model = PConvUNet()
    model.load_state_dict(torch.load('./trained_model.pth'))
    testDataset = util.build_dataset(TEST_IMG, TEST_MSK, isTrain=False)

    # evaluate(model, testDataset)
    show_visual_result(model, testDataset, "result")