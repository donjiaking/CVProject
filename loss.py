import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import torch.nn.functional as F



def gram_matrix(feature_matrix):
	batch, channel, h, w = feature_matrix.size()
	feature_matrix = feature_matrix.view(batch, channel, h*w)
	feature_matrix_t = feature_matrix.transpose(1, 2)

	# (batch, channel, h*w) x (batch, h*w, channel) -> (batch, channel, channel)
    # then multiply K_p=1/(channel*h*w), which is the normalization factor
	gram_mat = torch.bmm(feature_matrix, feature_matrix_t) / (channel*h*w)

	return gram_mat


"""
project images into higher level feature spaces using part of 
VGG16 network (pool1, pool2, pool3)
"""

class L1LossMultiScale(nn.Module):
    def __init__(self):
        super(L1LossMultiScale, self).__init__()
        self.weights = [1, 0.5, 0.25, 0.125]
        self.l1 = nn.L1Loss()
        self.downsample = nn.AvgPool2d(kernel_size = 2, count_include_pad=False)
        
    def forward(self, input, target):
        loss = 0
        for each_weight in self.weights:
            loss += each_weight * self.l1(input, target)
            input = self.downsample(input)
            target = self.downsample(target)
        return loss

class VGG16Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.pool1 = vgg16.features[:5]
        self.pool2 = vgg16.features[5:10]
        self.pool3 = vgg16.features[10:17]

        for i in range(1, 4):
            for param in getattr(self, 'pool{:d}'.format(i)).parameters():
                param.requires_grad = False

    # given an image, return the outputs of three layers as a list
    def forward(self, image):
        results = [image]
        for i in range(1, 4):
            func = getattr(self, 'pool{:d}'.format(i))
            # output of pool i-1 -> pool i -> output of pool i
            results.append(func(results[-1]))
        return results[1:]
        

class LossFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_extract = VGG16Extractor()
        self.l1 = nn.L1Loss()
        self.l1_multi_scale = L1LossMultiScale()




    def forward(self, input_img, mask, out, gt):
        comp_out = (gt * mask) + (out * (1 - mask))

        #### generate feature maps of comp_out, out, gt
        f_comp_out = self.vgg_extract(comp_out)
        f_out = self.vgg_extract(out)
        f_gt = self.vgg_extract(gt)
        
        #### compute loss values
        l_valid = self.l1(mask*out, mask*gt)
        l_hole = self.l1((1-mask)*out, (1-mask)*gt)

        l_perceptual = 0.0
        for p in range(len(f_out)):
            l_perceptual += self.l1(f_out[p], f_gt[p]) + self.l1(f_comp_out[p], f_gt[p])
        
        l_style = 0.0
        for p in range(len(f_out)):
            l_style += self.l1(gram_matrix(f_out[p]), gram_matrix(f_gt[p]))
            l_style += self.l1(gram_matrix(f_comp_out[p]), gram_matrix(f_gt[p]))

        l_tv = self.l1(comp_out[:,:,:,:-1], comp_out[:,:,:,1:]) + \
               self.l1(comp_out[:,:,:-1,:], comp_out[:,:,1:,:])
        
        l_l1multi = self.l1_multi_scale(out,gt)
        print(l_l1multi)
        total_loss = l_valid + 6*l_hole + 0.1*l_perceptual + 120*l_style + 0.1*l_tv + l_l1multi

        return total_loss