import torch
import torch.nn as nn
import torch.nn.functional as F

from pconv import PConv2d


class PConvUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # NOTE: the input image size is 256*256, so the number of encoder stage layers 
        # should be fixed to 7, the same for decoder stage

        ######################## Encoder (pconv_1 - pconv_7) ############################
        # 3x256x256->64x128x128
        self.pconv_1 = PConv2d(3, 64, has_bn=False, sample="down-7")

        # 64x128x128->128x64x64
        self.pconv_2 = PConv2d(64, 128, sample="down-5")

        # 128x64x64->256x32x32
        self.pconv_3 = PConv2d(128, 256, sample="down-5")

        # 256x32x32->512x16x16
        self.pconv_4 = PConv2d(256, 512, sample="down-3")

        # 512x16x16->512x8x8->512x4x4->512x2x2
        self.pconv_5 = PConv2d(512, 512, sample="down-3")
        self.pconv_6 = PConv2d(512, 512, sample="down-3")
        self.pconv_7 = PConv2d(512, 512, sample="down-3")

        ####################### Decoder (pconv_8 - pconv_14) ############################
        # 512x4x4(upsample output) + 512x4x4(pconv_6 output) = 1024x4x4->512x4x4
        self.pconv_8 = PConv2d(512+512, 512, nonlinearity="leaky")

		# 512x8x8(upsample output) + 512x8x8(pconv_5 output) = 1024x8x8->512x8x8
        self.pconv_9 = PConv2d(512+512, 512, nonlinearity="leaky")

		# 512x16x16(upsample output) + 512x16x16(pconv_4 output) = 1024x16x16->512x16x16
        self.pconv_10 = PConv2d(512+512, 512, nonlinearity="leaky")

        # 512x32x32(upsample output) + 256x32x32(pconv_3 output) = 768x32x32->256x32x32
        self.pconv_11 = PConv2d(512+256, 256, nonlinearity="leaky")

		# 256x64x64(upsample output) + 128x64x64(pconv_2 output) = 384x64x64->128x64x64
        self.pconv_12 = PConv2d(256+128, 128, nonlinearity="leaky")

		# 128x128x128(upsample output) + 64x128x128(pconv_1 output) = 192x128x128->64x128x128
        self.pconv_13 = PConv2d(128+64, 64, nonlinearity="leaky")

		# 64x256x256(upsample output) + 3x256x256(original image) = 67x256x256->3x256x256
        self.pconv_14 = PConv2d(64+3, 3, has_bn=False, nonlinearity="")
    

    def forward(self, x, m):
        ######################## Encoder (pconv_1 - pconv_7) ############################
        original_x, original_m = x, m
        out1_x, out1_m = self.pconv_1(x, m)
        out2_x, out2_m = self.pconv_2(out1_x, out1_m)
        out3_x, out3_m = self.pconv_3(out2_x, out2_m)
        out4_x, out4_m = self.pconv_4(out3_x, out3_m)
        out5_x, out5_m = self.pconv_5(out4_x, out4_m)
        out6_x, out6_m = self.pconv_6(out5_x, out5_m)
        out7_x, out7_m = self.pconv_7(out6_x, out6_m)

        ####################### Decoder (pconv_8 - pconv_14) ############################
        x = F.interpolate(out7_x, scale_factor=2)
        m = F.interpolate(out7_m, scale_factor=2)
        x = torch.cat([x, out6_x], dim=1)
        m = torch.cat([m, out6_m], dim=1)
        x, m = self.pconv_8(x, m)

        x = F.interpolate(x, scale_factor=2)
        m = F.interpolate(m, scale_factor=2)
        x = torch.cat([x, out5_x], dim=1)
        m = torch.cat([m, out5_m], dim=1)
        x, m = self.pconv_9(x, m)

        x = F.interpolate(x, scale_factor=2)
        m = F.interpolate(m, scale_factor=2)
        x = torch.cat([x, out4_x], dim=1)
        m = torch.cat([m, out4_m], dim=1)
        x, m = self.pconv_10(x, m)

        x = F.interpolate(x, scale_factor=2)
        m = F.interpolate(m, scale_factor=2)
        x = torch.cat([x, out3_x], dim=1)
        m = torch.cat([m, out3_m], dim=1)
        x, m = self.pconv_11(x, m)

        x = F.interpolate(x, scale_factor=2)
        m = F.interpolate(m, scale_factor=2)
        x = torch.cat([x, out2_x], dim=1)
        m = torch.cat([m, out2_m], dim=1)
        x, m = self.pconv_12(x, m)

        x = F.interpolate(x, scale_factor=2)
        m = F.interpolate(m, scale_factor=2)
        x = torch.cat([x, out1_x], dim=1)
        m = torch.cat([m, out1_m], dim=1)
        x, m = self.pconv_13(x, m)

        x = F.interpolate(x, scale_factor=2)
        m = F.interpolate(m, scale_factor=2)
        x = torch.cat([x, original_x], dim=1)
        m = torch.cat([m, original_m], dim=1)
        x, m = self.pconv_14(x, m)

        return x