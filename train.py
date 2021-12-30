import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import argparse
import os

from net import PConvUNet
from loss import LossFunc
from dataset import build_dataset
import util

##added 2021/12/27
device = torch.device('cuda')

def train(model, args):
    trainDataset = build_dataset(args.train_dir, args.mask_dir, isTrain=True)
    valDataset = build_dataset(args.val_dir, args.mask_dir, isTrain=False)
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False)

    criterion = LossFunc().to(device)
    optimizer = optim.Adam(filter(lambda x:x.requires_grad, model.parameters()), lr=args.init_lr)
    
    perform_dict_train = {"ssim":[], "psnr":[], "l1":[]}
    perform_dict_val = {"ssim":[], "psnr":[], "l1":[]}
    train_length = len(train_loader)

    print('Training Started!')
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        _adjust_lr(optimizer, args.init_lr, epoch) 

        ssim = psnr = l1 = 0.0
        mean_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            img, mask, gt = data
            img = img.to(device)
            mask = mask.to(device)
            gt = gt.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            output = model(img, mask)
            loss = criterion(img, mask, output, gt)
            loss.backward()
            optimizer.step()

            output, gt = util.unnormalize(output), util.unnormalize(gt)
            with torch.no_grad():
                ssim += util.ssim(output, gt)
                psnr += util.psnr(output, gt)
                l1 += util.l1loss(output, gt)
            mean_loss += loss.item()

            # print statistics after every 5 batches
            if((i+1) % 5 == 0):
                print('Epoch: [{}/{}][{}/{}] Loss {:.4f}'
                .format(epoch+1, args.epochs, i+1, len(train_loader), loss.item()))

        mean_loss /= train_length
        ssim /= train_length
        psnr /= train_length
        l1 /= train_length
        perform_dict_train['ssim'].append(ssim.item())
        perform_dict_train['psnr'].append(psnr.item())
        perform_dict_train['l1'].append(l1.item())

        ssim_val,psnr_val,l1_val = evaluate(model, val_loader)
        perform_dict_val['ssim'].append(ssim_val.item())
        perform_dict_val['psnr'].append(psnr_val.item())
        perform_dict_val['l1'].append(l1_val.item())

        # print statistics after one epoch
        print('Epoch [{}/{}] completed: Loss {:.4f}'.format(epoch+1, args.epochs, mean_loss))
        print("Performance on the training set: PSNR {:.3f} | SSIM {:.3f} | L1 {:.3f}"
             .format(psnr, ssim, l1))
        print("Performance on the validation set: PSNR {:.3f} | SSIM {:.3f} | L1 {:.3f}"
             .format(psnr_val, ssim_val, l1_val))
    
    util.save_model(model, args.model_dir)
    util.plot_performance(perform_dict_train, perform_dict_val, args.out_dir)
    print('Training Finished!')
    print('Training Time: {:.2f}'.format(time.time()-start_time))


"""
return ssim, psnr, l1 error measurements on the validation set
"""
def evaluate(model, val_loader):
    model.eval()
    ssim = psnr = l1 = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            img, mask, gt = data
            img = img.to(device)
            mask = mask.to(device)
            gt = util.unnormalize(gt.to(device))
            output = model(img, mask)
            output = util.unnormalize(output)
            ssim += util.ssim(output,gt)
            psnr += util.psnr(output,gt)
            l1 += util.l1loss(gt,output)

    val_length = len(val_loader)
    return ssim/val_length, psnr/val_length, l1/val_length


"""
decrease lr every `step_size` epochs by `decay` times
"""
def _adjust_lr(optimizer, init_lr, epoch_num, decay=0.6, step_size=1):
    lr = init_lr * (decay ** (epoch_num//step_size))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_lr', type=float, default=2e-3)
    parser.add_argument('--batch_size', type=int, default=4) #when in gtx960, the batch_size will be set to 4
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--train_dir', type=str, default="./dataset/train")
    parser.add_argument('--val_dir', type=str, default="./dataset/val")
    parser.add_argument('--mask_dir', type=str, default="./dataset/masks")
    parser.add_argument('--model_dir', type=str, default="./model")
    parser.add_argument('--out_dir', type=str, default="./result")
    args = parser.parse_args()

    model = PConvUNet().to(device)
    train(model, args)
