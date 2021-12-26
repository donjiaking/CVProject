import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from net import PConvUNet
from loss import LossFunc
import util

TRAIN_IMG = './dataset/train'
TRAIN_MSK = './dataset/masks'

EPOCHS_NUM = 1
BATCH_SIZE = 16

INITIAL_LR = 2e-4
# WEIGHT_DECAY = 1e-4

def train(model):
    trainDataset = util.build_dataset(TRAIN_IMG, TRAIN_MSK, isTrain=True)
    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = LossFunc()
    optimizer = optim.Adam(filter(lambda x:x.requires_grad, model.parameters()), lr=INITIAL_LR)
    
    print('Training Started')
    start_time = time.time()

    model.train()

    for epoch in range(EPOCHS_NUM):
        # _adjust_lr(optimizer, epoch+1) 

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            img, mask, gt = data

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
    
    torch.save(model.state_dict(), './trained_model.pth')
    print('Training Finished')
    print('Training Time: {:.2f}'.format(time.time()-start_time))


# adust lr every 20 epochs
def _adjust_lr(optimizer, epoch_num):
    lr = INITIAL_LR * (0.1 ** (epoch_num//20))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    model = PConvUNet()
    train(model)