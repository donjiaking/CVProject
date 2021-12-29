import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from piq.functional import gaussian_filter
from piq.utils import _validate_input, _reduce
from piq.ssim import _ssim_per_channel


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

def l1loss(x, y):
    return nn.L1Loss()(x, y)


def save_model(model, model_dir):
    if(not os.path.exists(model_dir)):
        os.mkdir(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))


def plot_performance(perform_dict_train, perform_dict_val, out_dir):
    x = range(1, len(perform_dict_train['ssim'])+1)

    for key in perform_dict_train.keys():
        name = key.upper()
        plt.plot(x,perform_dict_train[key],color='#d46b5f',label="training set")
        plt.plot(x,perform_dict_val[key],color='#50c3e6',label="validation set")
        plt.xlabel('epoch')
        plt.ylabel(name + " value")
        plt.title(name + " value" + " on the training/validation set")
        plt.legend()
        plt.text(list(x)[-1]-0.2,perform_dict_train[key][-1]+0.2,"{} = {:.3f}".format(name,perform_dict_train[key][-1]),color='#d46b5f')
        plt.text(list(x)[-1]-0.2,perform_dict_val[key][-1]+0.2,"{} = {:.3f}".format(name,perform_dict_val[key][-1]),color='#50c3e6')
        plt.scatter([list(x)[-1]], [perform_dict_train[key][-1]], s=25, c='#d46b5f')
        plt.scatter([list(x)[-1]], [perform_dict_val[key][-1]], s=25, c='#50c3e6')
        plt.savefig(os.path.join(out_dir,key))
        plt.clf()


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor([0.229, 0.224, 0.225]).to(device) + \
            torch.Tensor([0.485, 0.456, 0.406]).to(device)
    x = x.transpose(1, 3)

    return x


#### For testing
if __name__ == "__main__":
    perform_dict_train = {}
    perform_dict_train["ssim"] = [6,7,8,9]
    perform_dict_train["psnr"] = [6,7,8,10]
    perform_dict_train["l1"] = [15,20,20,20]
    perform_dict_val = {}
    perform_dict_val["ssim"] = [2,3,4,5]
    perform_dict_val["psnr"] = [5,6,7,8]
    perform_dict_val["l1"] = [8,9,0,2.366698]

    plot_performance(perform_dict_train, perform_dict_val, './result')