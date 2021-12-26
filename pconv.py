import torch
import torch.nn as nn
from torchvision import models

"""
Partial Convolution Layer used in PConvUNet
"""
class PConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, sample="none",
                                    has_bn=True, nonlinearity="relu") -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # if need to apply Batch Normalization
        if(has_bn):
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        # choose activation function
        if(nonlinearity == "relu"):
            self.activation = nn.ReLU()
        elif(nonlinearity == "leaky"):
            self.activation = nn.LeakyReLU()
        else:
            self.activation = None
        
        # different down-sampling cases
        if(sample == "down-7"):
            self.x_conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=7, stride=2, padding=3, bias=bias)
            self.m_conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=7, stride=2, padding=3, bias=False)
        elif(sample == "down-5"):
            self.x_conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=5, stride=2, padding=2, bias=bias)
            self.m_conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=5, stride=2, padding=2, bias=False)
        elif(sample == "down-3"):
            self.x_conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=2, padding=1, bias=bias)
            self.m_conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=2, padding=1, bias=False)
        else:  # no down-sample
            self.x_conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, bias=bias)
            self.m_conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, bias=False)
        
        # initialize kernel weight
        nn.init.kaiming_normal_(self.x_conv.weight, mode='fan_in', a=0)
        nn.init.constant_(self.m_conv.weight, 1.0)

        for param in self.m_conv.parameters():
            param.requires_grad = False
    
    def forward(self, x, m):
        x_conv = self.x_conv(torch.mul(x, m))

        with torch.no_grad():
            m_conv = self.m_conv(m)

        if self.x_conv.bias is not None:
            bias = self.x_conv.bias.view(1, -1, 1, 1).expand_as(x_conv)
        else:
            bias = torch.zeros_like(x_conv)
        
        # mask_ratio = sum(1) / sum(M)
        mask_ratio = (self.in_channels*self.x_conv.kernel_size[0]*self.x_conv.kernel_size[1])/(m_conv+1e-8)
        updated_mask = torch.clamp(m_conv, 0, 1)
        mask_ratio = torch.mul(mask_ratio, updated_mask)

        out = torch.mul(x_conv-bias, mask_ratio) + bias
        out = torch.mul(updated_mask, out)

        if(self.bn):
            out = self.bn(out)
        if(self.activation):
            out = self.activation(out)

        return out, updated_mask

            
        

