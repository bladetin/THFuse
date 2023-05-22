import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import Channel, Spatial

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last 

    def forward(self, x):
        x = x.cuda()
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.res = ResidualBlock(out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res(x)
        x = self.conv2(x)
        return x

class Fusenet(nn.Module):
    def __init__(self, input_nc=2):
        super(Fusenet, self).__init__()
        
        kernel_size = 1
        stride = 1

        # ---------------CNN--------------
        self.conv_in = ConvLayer(input_nc, 16, 3, 1)

        self.Conv_d = ConvLayer(16, 8, 3, 1)
        self.layers = nn.ModuleDict({
            'DenseConv1': ConvLayer(8, 8, 3, 1),
            'DenseConv2': ConvLayer(16, 8, 3, 1),
            'DenseConv3': ConvLayer(24, 8, 3, 1)
        })

        self.Conv1 = ConvLayer(16, 32, 3, 2)
        self.Conv2 = ConvLayer(32, 64, 3, 2)
        self.Conv3 = ConvLayer(64, 32, 3, 2)
        self.Upsample = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        
        # ----------------vit--------------
        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(4)
        self.down3 = nn.AvgPool2d(8)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_in1 = ConvLayer(input_nc, input_nc, kernel_size, stride)
        self.conv_out = ConvLayer(64, 1, kernel_size, stride, is_last=True)

        self.en0 = Encoder(64, 64, kernel_size, stride)
        self.en1 = Encoder(64, 64, kernel_size, stride)
        self.en2 = Encoder(64, 64, kernel_size, stride)
        self.en3 = Encoder(64, 64, kernel_size, stride)

        self.ctrans3 = Channel(size=32, embed_dim=128, patch_size=16, channel=64)
        self.strans3 = Spatial(size=256, embed_dim=1024*2, patch_size=4, channel=64)

    def forward(self, vi, ir):
        f0 = torch.cat([vi, ir], dim=1)
        x = self.conv_in1(f0)
        x = self.conv_in(x) 
        x_d = self.Conv_d(x) 

        for i in range(len(self.layers)):  
            out = self.layers['DenseConv' + str(i + 1)](x_d)
            x_d = torch.cat([x_d, out], 1)

        x_s = self.Conv1(x)  
        x_s = self.Conv2(x_s) 
        x_s = self.Conv3(x_s)  
        x_s = self.Upsample(x_s) 

        x0 = torch.cat([x_d, x_s], dim=1) 

        x0 = self.en0(x0) 
        x1 = self.en1(self.down1(x0)) 
        x2 = self.en2(self.down1(x1)) 
        x3 = self.en3(self.down1(x2)) 

        x3t = self.strans3(self.ctrans3(x3)) 
        x3m = x3t 
        x3r = x3 * x3m
        x2m = self.up1(x3m) 
        x2r = x2 * x2m 
        x1m = self.up1(x2m) + self.up2(x3m) 
        x1r = x1 * x1m 
        x0m = self.up1(x1m) + self.up2(x2m) + self.up3(x3m) 
        x0r = x0 * x0m 

        other =self.up3(x3r) + self.up2(x2r) + self.up1(x1r) + x0r 
        f1 = self.conv_out(other) 

        return f1
         
