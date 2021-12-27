import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import argparse
import os

from net import PConvUNet
from loss import LossFunc
import util

##added 2021/12/27
device = torch.device('cuda')


def train(model, args):
    trainDataset = util.build_dataset(args.img_dir, args.mask_dir, isTrain=True)
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False)

    criterion = LossFunc().to(device)
    optimizer = optim.Adam(filter(lambda x:x.requires_grad, model.parameters()), lr=args.init_lr)
    
    print('Training Started')
    start_time = time.time()

    model.train()

    for epoch in range(args.epochs):
        # _adjust_lr(optimizer, args.init_lr, epoch+1) 

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            img, mask, gt = data

            img = img.to(device)
            mask = mask.to(device)
            gt = gt.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(img, mask)
            loss = criterion(img, mask, output, gt)
            loss.backward()
            optimizer.step()

            # print statistics after every 1 batch
            if((i+1) % 1 == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.4f}'.format(
                       epoch+1, i+1, len(train_loader), loss.item()))
    
    if(not os.path.exists(args.out_dir)):
        os.mkdir(args.out_dir)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))
    print('Training Finished')
    print('Training Time: {:.2f}'.format(time.time()-start_time))


# adust lr every 20 epochs
def _adjust_lr(optimizer, init_lr, epoch_num):
    lr = init_lr * (0.1 ** (epoch_num//20))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=4) #when in gtx960, the batch_size will be set to 4
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--img_dir', type=str, default="./dataset/train")
    parser.add_argument('--mask_dir', type=str, default="./dataset/masks")
    parser.add_argument('--out_dir', type=str, default="./model")
    args = parser.parse_args()

    model = PConvUNet().to(device)
    train(model, args)