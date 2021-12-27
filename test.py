from numpy.core.fromnumeric import size
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import make_grid
from torchvision.utils import save_image
import time
import os
import argparse
from piq.functional import gaussian_filter
from piq.utils import _validate_input, _reduce
from piq.ssim import _ssim_per_channel

import util
from net import PConvUNet

##added 2021/12/27
device = torch.device('cuda')


"""
SSIM and PSNR function to evaluate the performance
    References:
        https://piq.readthedocs.io/en/latest/
"""
def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
        reduction: str = 'mean', full: bool = False, k1: float = 0.01, k2: float = 0.03):
    x = x.type(torch.float32)
    y = y.type(torch.float32)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    ssim_map, cs_map = _ssim_per_channel(x=x, y=y, kernel=kernel,  k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    ssim_val = _reduce(ssim_val, reduction)
    cs = _reduce(cs, reduction)

    if full:
        return [ssim_val, cs]

    return ssim_val

def psnr(x: torch.Tensor, y: torch.Tensor, 
         reduction: str = 'mean', convert_to_greyscale: bool = False):

    # Constant for numerical stability
    EPS = 1e-8

    if (x.size(1) == 3) and convert_to_greyscale:
        # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
        rgb_to_grey = torch.tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(x)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
        y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score: torch.Tensor = - 10 * torch.log10(mse + EPS)

    return _reduce(score, reduction)

"""
Evaluate the model given a dataset
"""
def evaluate(model, testDataset, args):
    test_loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

    print("Testing Started")  
    start_time = time.time()
    test_length = len(test_loader)
    model.eval()
    get_l1 = torch.nn.L1Loss()
    perform_dict = {}
    perform_dict["ssim"] = perform_dict["psnr"] = perform_dict["l1"] = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            
            # get the input
            img, mask, gt = data
            img = img.to(device)
            mask = mask.to(device)
            gt = util.unnormalize(gt.to(device))
            # get the output
            output = model(img, mask)
            output = util.unnormalize(output)
            # perform Function
            
            perform_dict["ssim"] += ssim(output,gt)
            perform_dict["psnr"] += psnr(output,gt)
            perform_dict["l1"] += get_l1(gt,output)
            print("Process: Batch " + str(i)+"/"+str(test_length) + ' PSNR：{:.4f}，SSIM：{:.4f}，L1：{:.4f}'.format(perform_dict["psnr"]/(i+1), perform_dict["ssim"]/(i+1), perform_dict["l1"]/(i+1)))
        #GET AVG
        perform_dict["ssim"]  = perform_dict["ssim"] / test_length
        perform_dict["psnr"] = perform_dict["psnr"] / test_length
        perform_dict["l1"] = perform_dict["l1"] / test_length
        print('AVG Value : PSNR：{}，SSIM：{}，L1：{}'.format(perform_dict["psnr"], perform_dict["ssim"], perform_dict["l1"]))

    print('Testing Finished')
    print('Testing Time: {:.2f}'.format(time.time()-start_time))

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

    print("Visualization Result successfully saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)#when in gtx960, the batch_size will be set to 4
    parser.add_argument('--img_dir', type=str, default="./dataset/val")
    parser.add_argument('--mask_dir', type=str, default="./dataset/masks")
    parser.add_argument('--model_dir', type=str, default="./model")
    parser.add_argument('--out_dir', type=str, default="./result")
    args = parser.parse_args()

    model = PConvUNet().to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth")))
    testDataset = util.build_dataset(args.img_dir, args.mask_dir, isTrain=False)

    evaluate(model, testDataset, args)
    show_visual_result(model, testDataset, args.out_dir)
