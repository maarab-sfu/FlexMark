import torch
import torchvision
import torch.nn as nn
import itertools
import numpy as np

import torch.nn.functional as F

import functools
import torch.utils.data as data

from PIL import Image, ImageEnhance
import os
import os.path

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_

from torchvision import models
from torchsummary import summary

import math
import cv2

from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as F_t
from random import randint, random
import skimage

import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



from tqdm import tqdm
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from itertools import chain

from itertools import cycle, compress


from math import exp
import re
from torch.nn.functional import conv2d

from skimage.metrics import peak_signal_noise_ratio, structural_similarity




class MyImageFolder(data.Dataset):

    def __init__(self, dir, transforms_=None, img_size=128, mask_size=64, length = 1000, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.length = 0
        self.image_list=[]
        for root, dirs, files in os.walk(dir):
            if mode == 'train':
                for name in files:
                    if name.endswith((".jpg","jpeg", ".png")):
                        path = os.path.join(dir, root, name)
                        self.length += 1
                        self.image_list.append(path)
                        if self.length == length:
                            return
            else:
                sorted_files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))
                for name in sorted_files:
                    if name.endswith((".jpg","jpeg", ".png")):
                        path = os.path.join(dir, root, name)
                        self.length += 1
                        self.image_list.append(path)
                        if self.length == length:
                            return


    def __getitem__(self, index):
        # print(self.image_list[index % len(self.image_list)])
        img = Image.open(self.image_list[index % len(self.image_list)]).convert('RGB')
        # print(img.shape)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_list)

"""Models"""




def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    
def sntconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].fill_(1.)  # Initialize scale to 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spectral_norm_flag = False):
        super(GenBlock, self).__init__()
        self.cond_bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        if spectral_norm_flag:
            self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.cond_bn2 = nn.BatchNorm2d(out_channels)
            self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.snconv2d1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.cond_bn2 = nn.BatchNorm2d(out_channels)
            self.snconv2d2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.snconv2d0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x
        x = self.cond_bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

def spatial_repeat(x, data):
    """
    This function takes a 5d tensor (with the same shape and dimension order
    as the input to Conv3d) and a 2d data tensor. For each element in the
    batch, the data vector is replicated spatially/temporally and concatenated
    to the channel dimension.
    Input: (N, C_{in}, H, W), (N, D)
    Output: (N, C_{in} + D, H, W)
    """
    N, D = data.size()
    N, _, H, W = x.size()
    x = torch.cat([
        x,
        data.view(N, D, 1, 1).expand(N, D, H, W)
    ], dim=1)
    return x

class Generator2(nn.Module):
    '''Generator for the watermarking of the image. secret message is an 'n x D' vector, and the image is of shape "n x 3 x img_size x img_size"'''

    def __init__(self, in_channel, g_conv_dim, alpha=0.01, deepmark_level = 1):
        super(Generator2, self).__init__()

        self.in_channel = in_channel
        self.g_conv_dim = g_conv_dim
        self.alpha = alpha
        self.deepmark_level = deepmark_level

        self.inc = DoubleConv(self.in_channel, self.g_conv_dim)
        self.down1 = Down(self.g_conv_dim, self.g_conv_dim * 2)
        self.self_attn1 = Self_Attn(g_conv_dim*2)
        self.down2 = Down(self.g_conv_dim * 2, self.g_conv_dim * 4)


        self.block1 = GenBlock(g_conv_dim*4, g_conv_dim*4)
        self.block2 = GenBlock(g_conv_dim*(4+2), g_conv_dim*2)

        self.block_s = nn.Conv2d(in_channels=3, out_channels=g_conv_dim*2, kernel_size=3, stride=1, padding=1)
        self.snconv2d1 = nn.Conv2d(in_channels=g_conv_dim*2, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(g_conv_dim*2, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.tanh = nn.Tanh()

        # Weight init
        # self.apply(init_weights)

    def forward(self, img, secret):
        # img: n x 3 x H x W | secret: n x 3 x H x W

        x0 = self.inc(img) # n x g_conv_dim x H x W
        x1 = self.down1(x0) # n x (g_conv_dim*2) x H/2 x W/2
        x1 = self.self_attn1(x1) # n x (g_conv_dim*2) x H/2 x W/2
        x2 = self.down2(x1) # n x (g_conv_dim*4) x H/4 x W/4
        
        x3 = self.block1(x2) # n x (g_conv_dim*4) x H/2 x W/2
        x33 = torch.cat([x1,x3], dim=1)
        x4 = self.block2(x33) # n x (g_conv_dim*2) x H x W
        x4 = self.bn(x4)                
        x4 = self.relu(x4)

        if self.deepmark_level == 1:
            secret = secret * 2.0 - 1.0
        secret = self.block_s(secret)
        x4 = x4 + self.alpha * secret

        x4 = self.snconv2d1(x4)         # n x 3 x 128 x 128
        out = self.tanh(x4)              # n x 3 x 128 x 128
        return out


class Generator1(nn.Module):
    """Generator."""
    def __init__(self, in_channel, g_conv_dim, alpha=0.01, deepmark_level = 1):
        super(Generator1, self).__init__()

        self.in_channel = in_channel
        self.g_conv_dim = g_conv_dim
        self.alpha = alpha
        self.deepmark_level = deepmark_level

        self.inc = DoubleConv(self.in_channel, self.g_conv_dim)
        self.down1 = Down(self.g_conv_dim, self.g_conv_dim * 2)
        self.down2 = Down(self.g_conv_dim * 2, self.g_conv_dim * 4)
        self.down3 = Down(self.g_conv_dim * 4, self.g_conv_dim * 8)
        self.self_attn1 = Self_Attn(g_conv_dim*8)
        self.down4 = Down(self.g_conv_dim * 8, self.g_conv_dim * 16)
        self.down5 = Down(self.g_conv_dim * 16, self.g_conv_dim * 16)

        self.block1 = GenBlock(g_conv_dim*16, g_conv_dim*16)
        self.block2 = GenBlock(g_conv_dim*(16+16), g_conv_dim*8)
        self.block3 = GenBlock(g_conv_dim*(8+8), g_conv_dim*4)
        self.self_attn2 = Self_Attn(g_conv_dim*4)
        self.block4 = GenBlock(g_conv_dim*(4+4), g_conv_dim*2)
        self.block5 = GenBlock(g_conv_dim*(2+2), g_conv_dim)

        self.block_s = nn.Conv2d(in_channels=3, out_channels=g_conv_dim, kernel_size=3, stride=1, padding=1)
        self.snconv2d1 = nn.Conv2d(in_channels=g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.tanh = nn.Tanh()

        # Weight init
        # self.apply(init_weights)

    def forward(self, img, secret):
        # img: n x 3 x H x W | secret: n x 3 x H x W

        x0 = self.inc(img) # n x g_conv_dim x H x W
        x1 = self.down1(x0) # n x (g_conv_dim*2) x H/2 x W/2
        x2 = self.down2(x1) # n x (g_conv_dim*4) x H/4 x W/4
        x3 = self.down3(x2) # n x (g_conv_dim*8) x H/8 x W/8

        x3 = self.self_attn1(x3) # n x (g_conv_dim*8) x H/8 x W/8

        x4 = self.down4(x3) # n x (g_conv_dim*16) x H/16 x W/16
        x5 = self.down5(x4) # n x (g_conv_dim*16) x H/32 x W/32

        x6 = self.block1(x5) # n x (g_conv_dim*16) x H/16 x W/16
        x66 = torch.cat([x4,x6], dim=1)
        x7 = self.block2(x66) # n x (g_conv_dim*8) x H/8 x W/8
        x77 = torch.cat([x3,x7], dim=1)
        x8 = self.block3(x77) # n x (g_conv_dim*4) x H/4 x W/4

        x8 = self.self_attn2(x8)

        x88 = torch.cat([x2,x8], dim=1)
        x9 = self.block4(x88) # n x (g_conv_dim*2) x H/2 x W/2
        x99 = torch.cat([x1,x9], dim=1)
        x10 = self.block5(x99)    # n x g_conv_dim  x H x W
        x10 = self.bn(x10)                
        x10 = self.relu(x10)

        if self.deepmark_level == 1:
            secret = secret * 2.0 - 1.0
        secret = self.block_s(secret)
        x10 = x10 + self.alpha * secret

        x10 = self.snconv2d1(x10)         # n x 3 x 128 x 128
        out = self.tanh(x10)              # n x 3 x 128 x 128
        return out




class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim):
        super(Discriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.opt_block1 = DiscOptBlock(3, d_conv_dim)
        self.block1 = DiscBlock(d_conv_dim, d_conv_dim*2)
        self.self_attn = Self_Attn(d_conv_dim*2)
        self.block2 = DiscBlock(d_conv_dim*2, d_conv_dim*4)
        self.block3 = DiscBlock(d_conv_dim*4, d_conv_dim*8)
        self.block4 = DiscBlock(d_conv_dim*8, d_conv_dim*16)
        self.block5 = DiscBlock(d_conv_dim*16, d_conv_dim*16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=d_conv_dim*16, out_features=1)
        self.sigmoid = nn.Sigmoid()
        # self.sn_embedding1 = sn_embedding(num_classes, d_conv_dim*16)

        # xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, x):
        # n x 3 x 128 x 128
        h0 = self.opt_block1(x) # n x d_conv_dim   x 64 x 64
        h1 = self.block1(h0)    # n x d_conv_dim*2 x 32 x 32
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 32 x 32
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 16 x 16
        h3 = self.block3(h2)    # n x d_conv_dim*8 x  8 x  8
        h4 = self.block4(h3)    # n x d_conv_dim*16 x 4 x  4
        h5 = self.block5(h4, downsample=False)  # n x d_conv_dim*16 x 4 x 4
        h5 = self.relu(h5)              # n x d_conv_dim*16 x 4 x 4
        h6 = torch.sum(h5, dim=[2,3])   # n x d_conv_dim*16
        output1 = torch.squeeze(self.snlinear1(h6)) # n
        output = self.sigmoid(output1)
        # Projection
        # h_labels = self.sn_embedding1(labels)   # n x d_conv_dim*16
        # proj = torch.mul(h6, h_labels)          # n x d_conv_dim*16
        # output2 = torch.sum(proj, dim=[1])      # n
        # Out
        # output = output1 + output2              # n
        return output

###############################
#           Decoder           #
###############################

class Decoder1(nn.Module):
    def __init__(self, input_size=2):
        super(Decoder1, self).__init__()

        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # Compute the sizes of the CNN layers based on the input size
        self.conv3_size = input_size  // 2 

        self.fc = nn.Linear(32 * self.conv3_size * self.conv3_size, 256)
        self.relu4 = nn.ReLU()
        self.output = nn.Linear(256, 8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply the CNN layers to the input image
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)

        # Flatten the output of the CNN to be passed through the fully connected layers
        x = x.view(-1, 32 * self.conv3_size * self.conv3_size)
        x = self.fc(x)
        x = self.relu4(x)
        x = self.output(x)

        # Clamp the output to be between 0 and 255, and cast to uint8
        x = self.sigmoid(x)

        return x


class Decoder2(nn.Module):
    def __init__(self, in_channel, g_conv_dim, img_size = 64):
        super(Decoder2, self).__init__()
        
        self.in_channel = in_channel
        self.g_conv_dim = g_conv_dim
        self.img_size = img_size

        self.inc = DoubleConv(self.in_channel, self.g_conv_dim)
        self.down1 = Down(self.g_conv_dim, self.g_conv_dim * 2)
        self.self_attn1 = Self_Attn(g_conv_dim*2)
        self.down2 = Down(self.g_conv_dim * 2, self.g_conv_dim * 4)
        
        self.fc1 = nn.Linear((self.g_conv_dim * 4) * int(self.img_size//4) * int(self.img_size//4), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)
        self.sigmoid = nn.Sigmoid()

        

    def forward(self, img):
        # img: n x 3 x H x W
        # _,_,H,W = img.shape
        # print(img.shape)
        x0 = self.inc(img) # n x g_conv_dim x H x W
        x1 = self.down1(x0) # n x (g_conv_dim*2) x H/2 x W/2
        x1 = self.self_attn1(x1) # n x (g_conv_dim*2) x H/2 x W/2
        x2 = self.down2(x1) # n x (g_conv_dim*4) x H/4 x W/4
        
        # print((self.g_conv_dim * 8) * int(self.img_size//8) * int(self.img_size//8))
        x4 = x2.reshape(-1,(self.g_conv_dim * 4) * int(self.img_size//4) * int(self.img_size//4)) # n x (g_conv_dim*8*H/8*W/8)
        # print(x4.shape)
        x5 = self.fc1(x4) # n x 120
        x6 = F.relu(self.fc2(x5)) # n x 84
        out = self.fc3(x6) # n x 8
        out = self.sigmoid(out)
          
        return out

"""Differentiable JPEG Approximation"""





y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))
#
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.


class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """
    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        #
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
    #    result = torch.from_numpy(result)
        result.view(image.shape)
        return result



class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """
    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    """
    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)
    

class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor =  nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float() )
        
    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, rounding, factor=1):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = y_table

    def forward(self, image):
        image = image.float() / (self.y_table * self.factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, rounding, factor=1):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.c_table = c_table

    def forward(self, image):
        image = image.float() / (self.c_table * self.factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """
    def __init__(self, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )
        self.c_quantize = c_quantize(rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(rounding=rounding, factor=factor)

    def forward(self, image):
        y, cb, cr = self.l1(image*255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp)
            else:
                comp = self.y_quantize(comp)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


class y_dequantize(nn.Module):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width
    """
    def __init__(self, factor=1):
        super(y_dequantize, self).__init__()
        self.y_table = y_table
        self.factor = factor

    def forward(self, image):
        return image * (self.y_table * self.factor)


class c_dequantize(nn.Module):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width
    """
    def __init__(self, factor=1):
        super(c_dequantize, self).__init__()
        self.factor = factor
        self.c_table = c_table

    def forward(self, image):
        return image * (self.c_table * self.factor)


class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self):
        super(block_merging, self).__init__()
        
    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height//k, width//k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input: 
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """
    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)
        
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """
    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()

        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        #result = torch.from_numpy(result)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)


class decompress_jpeg(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """
    def __init__(self, height, width, rounding=torch.round, factor=1):
        super(decompress_jpeg, self).__init__()
        self.c_dequantize = c_dequantize(factor=factor)
        self.y_dequantize = y_dequantize(factor=factor)
        self.idct = idct_8x8()
        self.merging = block_merging()
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()
        
        self.height, self.width = height, width
        
    def forward(self, y, cb, cr):
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k])
                height, width = int(self.height/2), int(self.width/2)                
            else:
                comp = self.y_dequantize(components[k])
                height, width = self.height, self.width                
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = torch.min(255*torch.ones_like(image),
                          torch.max(torch.zeros_like(image), image))
        return image/255






class DiffJPEG(nn.Module):
    def __init__(self, height=128, width=128, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, noised_and_cover):
        '''
        '''
        noised_image = noised_and_cover[0]
        y, cb, cr = self.compress(noised_image)
        noised_and_cover[0] = self.decompress(y, cb, cr)
        return noised_and_cover

"""noise layers"""





def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, noised_and_cover):
        return noised_and_cover


class GaussianNoise(nn.Module):
    def __init__(self, Standard_deviation):
        super(GaussianNoise, self).__init__()
        self.Standard_deviation = Standard_deviation

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy()
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(batch_encoded_image.shape[0]):
            encoded_image = batch_encoded_image[idx]
            noise_image = skimage.util.random_noise(encoded_image, mode= 'gaussian',clip = False, var = (self.Standard_deviation) ** 2 )
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = 2*batch_noise_image - 1
        return noised_and_cover


class SaltPepper(nn.Module):
    def __init__(self,Amount):
        super(SaltPepper, self).__init__()
        self.Amount = Amount

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch_encoded_image = noised_image.cpu().detach().numpy()
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(batch_encoded_image.shape[0]):
            encoded_image = batch_encoded_image[idx]
            noise_image = skimage.util.random_noise(encoded_image, mode='s&p', amount = self.Amount)
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = batch_noise_image
        return noised_and_cover

class GaussianBlur(nn.Module):
    def __init__(self, sigma):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(batch_encoded_image.shape[0]):
            encoded_image = batch_encoded_image[idx]
            noise_image = cv2.GaussianBlur(src = encoded_image,ksize = (0,0), sigmaX = self.sigma,sigmaY = self.sigma)
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)/255
        return noised_and_cover

class MedianFilter(nn.Module):
    def __init__(self, kernel = 7):
        super(MedianFilter, self).__init__()
        self.kernel = kernel

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(batch_encoded_image.shape[0]):
            encoded_image = batch_encoded_image[idx]
            noise_image = cv2.medianBlur(encoded_image, self.kernel)
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)/255
        return noised_and_cover

class AverageFilter(nn.Module):
    def __init__(self, kernel = 5):
        super(AverageFilter, self).__init__()
        self.kernel = kernel

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(batch_encoded_image.shape[0]):
            encoded_image = batch_encoded_image[idx]
            noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)/255
        return noised_and_cover




class DropOut(nn.Module):
    def __init__(self, prob):
        super(DropOut, self).__init__()
        self.prob = prob

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        mask = np.random.choice([0.0, 1.0], noised_image.shape[2:], p=[1 - self.prob, self.prob])
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        mask_tensor = mask_tensor.expand_as(noised_image)
        noised_image = noised_image * mask_tensor + cover_image * (1 - mask_tensor)
        noised_and_cover[0] = noised_image
        return noised_and_cover




class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        """
        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)
        crop_mask = torch.zeros_like(noised_image)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)
        crop_mask[:, :, h_start:h_end, w_start:w_end] = 1
        noised_and_cover[0] = noised_image * crop_mask
        return noised_and_cover

class CropOut(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        super(CropOut, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]

        cropout_mask = torch.zeros_like(noised_image)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        # noised_and_cover[0] = noised_image * (1-cropout_mask) + cover_image * cropout_mask
        noised_and_cover[0] = noised_image * (1-cropout_mask) + 0 * cropout_mask
        return  noised_and_cover

class PixelElimination(nn.Module):
    def __init__(self, pixel_ratio):
        super(PixelElimination, self).__init__()
        self.pixel_ratio = pixel_ratio

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        _,_,H,W = noised_image.shape

        elimination_mask = torch.ones_like(noised_image)

        idx_H = np.random.randint(H, size=(int(self.pixel_ratio*H)))
        idx_W = np.random.randint(W, size=(int(self.pixel_ratio*W)))

        elimination_mask[:, :, :, idx_W] = 0
        elimination_mask[:, :, idx_H, :] = 0

        noised_and_cover[0] = noised_image * elimination_mask
        return noised_and_cover

class AdjustHue(nn.Module):
    def __init__(self, hue_factor):
        super(AdjustHue, self).__init__()
        self.hue_factor = hue_factor
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F_t.adjust_hue(noised_image, self.hue_factor)
        return  noised_and_cover

class AdjustSaturation(nn.Module):
    def __init__(self, sat_factor):
        super(AdjustSaturation, self).__init__()
        self.sat_factor = sat_factor
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F_t.adjust_saturation(noised_image, self.sat_factor)
        return  noised_and_cover

class AdjustBrightness(nn.Module):
    def __init__(self, bri_factor):
        super(AdjustBrightness, self).__init__()
        self.bri_factor = bri_factor
    
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = (noised_image+1)/2
        batch_encoded_image = [ToPILImage()(x_) for x_ in noised_image]
        # batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        # batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(len(batch_encoded_image)):
            encoded_image = batch_encoded_image[idx]
            enhancer = ImageEnhance.Brightness(encoded_image)
            noise_image = enhancer.enhance(self.bri_factor)
            noise_image = ToTensor()(noise_image).type(torch.FloatTensor).cuda()
            # noise_image = (noise_image*2-1).type(torch.FloatTensor).cuda()
            # noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            # noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)
        return noised_and_cover

class AdjustContrast(nn.Module):
    def __init__(self, con_factor):
        super(AdjustContrast, self).__init__()
        self.con_factor = con_factor

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = (noised_image+1)/2
        batch_encoded_image = [ToPILImage()(x_) for x_ in noised_image]
        # batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        # batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(len(batch_encoded_image)):
            encoded_image = batch_encoded_image[idx]
            enhancer = ImageEnhance.Contrast(encoded_image)
            noise_image = enhancer.enhance(self.con_factor)
            noise_image = ToTensor()(noise_image).type(torch.FloatTensor).cuda()
            # noise_image = (noise_image*2-1).type(torch.FloatTensor).cuda()
            # noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            # noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)
        return noised_and_cover

class AdjustColor(nn.Module):
    def __init__(self, col_factor):
        super(AdjustColor, self).__init__()
        self.col_factor = col_factor
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = (noised_image+1)/2
        batch_encoded_image = [ToPILImage()(x_) for x_ in noised_image]
        # batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        # batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(len(batch_encoded_image)):
            encoded_image = batch_encoded_image[idx]
            enhancer = ImageEnhance.Color(encoded_image)
            noise_image = enhancer.enhance(self.col_factor)
            noise_image = ToTensor()(noise_image).type(torch.FloatTensor).cuda()
            # noise_image = (noise_image*2-1).type(torch.FloatTensor).cuda()
            # noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            # noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)
        return noised_and_cover

class AdjustSharpness(nn.Module):
    def __init__(self, sha_factor):
        super(AdjustSharpness, self).__init__()
        self.sha_factor = sha_factor
    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = (noised_image+1)/2
        batch_encoded_image = [ToPILImage()(x_) for x_ in noised_image]
        # batch_encoded_image = ((noised_image+1)/2).cpu().detach().numpy() * 255
        # batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
        for idx in range(len(batch_encoded_image)):
            encoded_image = batch_encoded_image[idx]
            enhancer = ImageEnhance.Sharpness(encoded_image)
            noise_image = enhancer.enhance(self.sha_factor)
            noise_image = ToTensor()(noise_image).type(torch.FloatTensor).cuda()
            # noise_image = (noise_image*2-1).type(torch.FloatTensor).cuda()
            # noise_image = cv2.blur(encoded_image, (self.kernel, self.kernel))
            # noise_image = torch.from_numpy(noise_image.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if (idx == 0):
                batch_noise_image = noise_image.unsqueeze(0)
            else:
                batch_noise_image = torch.cat((batch_noise_image, noise_image.unsqueeze(0)), 0)  # batch*H*W*C
        batch_noise_image = Variable(batch_noise_image, requires_grad=True).cuda()  # batch*C*H*W
        noised_and_cover[0] = (2*batch_noise_image - 1)
        return noised_and_cover



def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v= torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)

class Compression(nn.Module):
    """
    This uses the DCT to produce a differentiable approximation of JPEG compression.
    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, yuv=False, min_pct=0.0, max_pct=0.5):
        super(Compression, self).__init__()
        self.yuv = yuv
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, y):
        z = y[0]
        N, _, H, W = z.size()

        H = int(z.size(2) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        W = int(z.size(3) * (random() * (self.max_pct - self.min_pct) + self.min_pct))

        if self.yuv:
            z = torch.stack([
                (0.299 * z[:, 2, :, :] +
                 0.587 * z[:, 1, :, :] +
                 0.114 * z[:, 0, :, :]),
                (- 0.168736 * z[:, 2, :, :] -
                 0.331264 * z[:, 1, :, :] +
                 0.500 * z[:, 0, :, :]),
                (0.500 * z[:, 2, :, :] -
                 0.418688 * z[:, 1, :, :] -
                 0.081312 * z[:, 0, :, :]),
            ], dim=1)

        z = dct_3d(z)

        if H > 0:
            z[:, :, -H:, :] = 0.0

        if W > 0:
            z[:, :, :, -W:] = 0.0

        z = idct_3d(z)

        if self.yuv:
            z = torch.stack([
                (1.0 * z[:, 0, :, :] +
                 1.772 * z[:, 1, :, :] +
                 0.000 * z[:, 2, :, :]),
                (1.0 * z[:, 0, :, :] -
                 0.344136 * z[:, 1, :, :] -
                 0.714136 * z[:, 2, :, :]),
                (1.0 * z[:, 0, :, :] +
                 0.000 * z[:, 1, :, :] +
                 1.402 * z[:, 2, :, :]),
            ], dim=1)
        y[0]= z
        return y

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



#################################
### Compression Approximation ###
#################################

class UNet(nn.Module):
    def __init__(self, n_channels = 3 , n_classes = 3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, y):
        x = y[0]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        y[0] = logits
        return y



def transform(tensor, target_range):
    source_min = tensor.min()
    source_max = tensor.max()

    # normalize to [0, 1]
    tensor_target = (tensor - source_min)/(source_max - source_min)
    # move to target range
    tensor_target = tensor_target * (target_range[1] - target_range[0]) + target_range[0]
    return tensor_target


class Quantization(nn.Module):
    def __init__(self, device=None):
        super(Quantization, self).__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.min_value = 0.0
        self.max_value = 255.0
        self.N = 10
        self.weights = torch.tensor([((-1) ** (n + 1)) / (np.pi * (n + 1)) for n in range(self.N)]).to(device)
        self.scales = torch.tensor([2 * np.pi * (n + 1) for n in range(self.N)]).to(device)
        for _ in range(4):
            self.weights.unsqueeze_(-1)
            self.scales.unsqueeze_(-1)


    def fourier_rounding(self, tensor):
        shape = tensor.shape
        z = torch.mul(self.weights, torch.sin(torch.mul(tensor, self.scales)))
        z = torch.sum(z, dim=0)
        return tensor + z


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = transform(noised_image, (0, 255))
        # noised_image = noised_image.clamp(self.min_value, self.max_value).round()
        noised_image = self.fourier_rounding(noised_image.clamp(self.min_value, self.max_value))
        noised_image = transform(noised_image, (noised_and_cover[0].min(), noised_and_cover[0].max()))
        return [noised_image, noised_and_cover[1]]

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_range, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)

        return noised_and_cover



########################
#   Distortion Layer   #
########################

class DistortionLayer(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, infoDict, device, phase = "train"):
        super(DistortionLayer, self).__init__()
        self.noise_layers = [Identity()]
        self.phase = phase
        # self.noise_layers = JpegCompression(device)
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'Compression':
                    # self.noise_layers.append(DiffJPEG().to(device))
                    net = UNet()
                    checkpoints = torch.load("C:/Users/maarab/Forensics/Stegano/" + "JPEG_model_%d.pth"% (95))
                    net.to(device)
                    net.load_state_dict(checkpoints['net_state_dict'])
                    net.eval()
                    self.noise_layers.append(net)
                elif layer == 'WebP':
                    net = UNet()
                    checkpoints = torch.load("C:/Users/maarab/Forensics/Stegano/" + "WebP_model_%d.pth"% (95))
                    net.to(device)
                    net.load_state_dict(checkpoints['net_state_dict'])
                    net.eval()
                    self.noise_layers.append(net)
                elif layer == 'JPEG2000':
                    net2 = UNet()
                    checkpoints = torch.load("C:/Users/maarab/Forensics/Stegano/" + "JPEG2000_model_%d.pth"% (95))
                    net2.to(device)
                    net2.load_state_dict(checkpoints['net_state_dict'])
                    net2.eval()
                    self.noise_layers.append(net2)
                elif layer == 'CropOut':
                    self.noise_layers.append(CropOut(infoDict['cropout_ratio'], infoDict['cropout_ratio']))
                elif layer == 'Crop':
                    self.noise_layers.append(Crop(infoDict['crop_ratio'], infoDict['crop_ratio']))
                elif layer == 'PixelElimination':
                    self.noise_layers.append(PixelElimination(infoDict['pixel_ratio']))
                elif layer == 'DropOut':
                    self.noise_layers.append(DropOut(infoDict['prob']))
                elif layer == 'Quantization':
                    self.noise_layers.append(Quantization())
                elif layer == 'GaussianBlur':
                    self.noise_layers.append(GaussianBlur(infoDict['sigma']))
                elif layer == 'MedianFilter':
                    self.noise_layers.append(MedianFilter(infoDict['med_kernel']))
                elif layer == 'AverageFilter':
                    self.noise_layers.append(AverageFilter(infoDict['ave_kernel']))
                elif layer == 'GaussianNoise':
                    self.noise_layers.append(GaussianNoise(infoDict['Standard_deviation']))
                elif layer == 'SaltPepper':
                    self.noise_layers.append(SaltPepper(infoDict['Amount']))
                elif layer == 'AdjustHue':
                    self.noise_layers.append(AdjustHue(infoDict['hue_factor']))
                elif layer == 'AdjustSaturation':
                    self.noise_layers.append(AdjustSaturation(infoDict['sat_factor']))
                elif layer == 'AdjustColor':
                    self.noise_layers.append(AdjustColor(infoDict['col_factor']))
                elif layer == 'AdjustContrast':
                    self.noise_layers.append(AdjustContrast(infoDict['con_factor']))
                elif layer == 'AdjustSharpness':
                    self.noise_layers.append(AdjustSharpness(infoDict['sha_factor']))
                elif layer == 'AdjustBrightness':
                    self.noise_layers.append(AdjustBrightness(infoDict['bri_factor']))
                else:
                    raise ValueError(f'Wrong layer placeholder string in DistortionLayer.__init__().'
                                     f' Expected distortion layers but got {layer} instead')
            else:
                self.noise_layers.append(layer)

    def forward(self, encoded_and_cover):
        if self.phase == "train":
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
            return random_noise_layer(encoded_and_cover)

        elif self.phase == "test":
            output = encoded_and_cover
            for layer in self.noise_layers:
                output = layer(output)
            return output

"""Network"""

# Commented out IPython magic to ensure Python compatibility.
"""
Inpainting using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 context_encoder.py'
"""



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    parser.add_argument("--n_epochs", type=int, default=101, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--dataset_name", type=str, default="COCO-128bits", help="name of the dataset")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")

    parser.add_argument("--deepmark_level", type=int, default = 1, help="The flexibility of the deepmark. '1': is for the most robust case, in which a 64, or 128 bit string is embedded inside the video frame/ image. '2': is for content protection")
    parser.add_argument("--data_dim", type=int, default = 128, help="The dimension of the message.")
    parser.add_argument("--sec_ratio", type=int, default = 2, help="The ratio of the secret image to the original image. img/sec_ratio^2" )
    parser.add_argument("--continue_train", type=int, default = 50, help="continue the training")
    parser.add_argument("--use_noise", type=bool, default = True, help="Use distortion layer between the encoder and decoder")
    parser.add_argument("--phase", type = str, default = "extract", help="train or test or extract or embed")
    parser.add_argument("--bit_redundancy", type = int, default = 7, help="the number of redundant bits in each 8 bits")

    opt = parser.parse_args()
    # print(opt)

    
    if opt.deepmark_level == 1:
        opt.sec_ratio = opt.img_size//int(2**((math.log2(opt.data_dim)-3)//2))
        print(opt.sec_ratio)
    DATA_DIR = "C:/Users/maarab/Forensics/Datasets/COCO"

    if opt.deepmark_level == 1:
        DIR = "C:/Users/maarab/Forensics/Stegano/"
        
    os.makedirs(DIR + "images/", exist_ok=True)
    os.makedirs(DIR + "saved_models/%s/" % (opt.dataset_name), exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def dec2bin(x, bits=8):
        mask = 2**torch.arange(bits-1,-1,-1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

    def bin2dec(b, bits=8):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)

    def make_pair(frames, data_dim, use_bit_inverse=True):

        message_bits = math.log2(data_dim)
        # print(message_bits)
        message_bytes = message_bits - 3
        # print(message_bytes)
        if message_bytes % 2 == 0:
            message_shape = int(2 ** (message_bytes // 2))
            # print(message_shape)
        #else: redundant bits
        secret = torch.zeros(frames.size())
        message = torch.zeros(frames.size(0),1, message_shape, message_shape)
        for aa in range(message_shape):
            for bb in range(message_shape):
                rand_message = torch.rand(frames.size(0),1, 1, 1)
                message[:,0,aa,bb] = rand_message.squeeze()
                pad_size = int(frames.shape[-1]//message_shape)
                one_channel = nn.ReplicationPad2d((pad_size-1, 0, pad_size-1, 0))(rand_message)
                secret[:, :, aa*pad_size:(aa+1)*pad_size, bb*pad_size:(bb+1)*pad_size] = torch.cat([one_channel, one_channel, one_channel], dim=1)
        # Normalize "secret"
        secret = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(secret)
        message = dec2bin(((message)*255).int())
        message = message.reshape((frames.size(0), message_shape*message_shape* 8))
        return frames, secret, message


    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv2") != -1:
            xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("Linear") != -1:
            xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    # cover_loss = torch.nn.L1Loss()
    cover_loss = torch.nn.MSELoss()
    if opt.deepmark_level == 1:
        # message_loss = nn.BCEWithLogitsLoss()
        message_loss = nn.BCELoss()
    elif opt.deepmark_level == 2:
        perceptual_loss = VGGPerceptualLoss()
    # bce = torch.nn.BCEWithLogitsLoss()
    bce = torch.nn.BCELoss()
    noises = []
    # noises = ['Compression', 'CropOut', 'DropOut', 'GaussianBlur', 'GaussianNoise', 'SaltPepper', 'AdjustHue', 'AdjustSaturation', 'AdjustBrightness','AdjustContrast','AdjustColor','AdjustSharpness','JPEG2000', 'WebP', 'MedianFilter', 'AverageFilter', 'PixelElimination']
    # noises = ['Compression','GaussianBlur', 'GaussianNoise']
    # noises = ['MedianFilter']
    # noises = ['WebP', 'CropOut', 'SaltPepper', 'AverageFilter', 'AdjustHue']
    # noises = ['Compression', 'GaussianBlur', 'AdjustSaturation', 'AverageFilter', 'MedianFilter']
    # noises = ['CropOut', 'GaussianNoise', 'SaltPepper', 'DropOut', 'PixelElimination']

    noises = ['Compression']

    #########
    # Known #
    #########

    # noises = ['Quantization', 'DropOut', 'GaussianNoise', 'SaltPepper']
    # noises = ['Compression','CropOut', 'DropOut', 'GaussianBlur', 'Crop']
    # noises = ['Compression', 'JPEG2000', 'WebP']
    # noises = ['GaussianBlur','MedianFilter', 'AverageFilter']
    # noises = ['Crop', 'CropOut', 'PixelElimination']
    # noises = ['AdjustBrightness','AdjustContrast','AdjustColor','AdjustSharpness']

    ## Mixed ##

    # noises = ['SaltPepper', 'WebP', 'GaussianBlur', 'PixelElimination','AdjustContrast']
    # noises = ['GaussianNoise', 'CropOut', 'GaussianBlur', 'Compression', 'DropOut']

    ###########
    # Unknown #
    ###########
    # noises = ['SaltPepper']
    # noises = ['WebP']
    # noises = ['GaussianBlur']
    # noises = ['PixelElimination']
    # noises = ['AdjustContrast']

    # noises = ['GaussianNoise']

    # noises = ['DropOut']
    # noises = ['CropOut']

    # noises = ['Compression']

    # infoDict = {'pixel_ratio': 0.1 ,'cropout_ratio': (0.2,0.4),'crop_ratio':(0.03,0.04), 'prob': 0.3, 'sigma':2.0, 'Standard_deviation':0.03, 'Amount': 0.05, 'hue_factor':0.1, 'sat_factor':1.5, 'bri_factor':1.5,'con_factor':0.66,'sha_factor':1.5,'col_factor':1.5, 'ave_kernel':3, 'med_kernel':5} 
    infoDict = {'pixel_ratio': 0.1 ,'cropout_ratio': (0.2,0.4),'crop_ratio':(0.03,0.04), 'prob': 0.3, 'sigma':2.0, 'Standard_deviation':0.03, 'Amount': 0.05, 'hue_factor':0.05, 'sat_factor':1.5, 'bri_factor':1.5,'con_factor':0.66,'sha_factor':1.5,'col_factor':1.5, 'ave_kernel':3, 'med_kernel':5} 
    # infoDict = {'pixel_ratio': 0.2 ,'cropout_ratio': (0.4,0.5),'crop_ratio':(0.03,0.04), 'prob': 0.5, 'sigma':4.0, 'Standard_deviation':0.04, 'Amount': 0.07, 'hue_factor':0.05, 'sat_factor':2.0, 'bri_factor':2.0,'con_factor':2.0,'sha_factor':2.0,'col_factor':2.0, 'ave_kernel':7, 'med_kernel':7} 
 


    if opt.use_noise and opt.phase == "train": 
        distortion_layers = DistortionLayer(noises, infoDict, device)
        distortion_layers.to(device)
    elif opt.use_noise and opt.phase == "test":
        distortion_layers = DistortionLayer(noises, infoDict, device)
        distortion_layers.to(device)
    else:
        noises = []
        infoDict = {}
        distortion_layers = DistortionLayer(noises, infoDict, device)
        distortion_layers.to(device)

    # Initialize encoder, decoder, and discriminator
    encoder = Generator1(in_channel = 6, g_conv_dim = 64, alpha = 0.01)
    discriminator = Discriminator(d_conv_dim = 64)
    if opt.deepmark_level == 1:
        decoder = Decoder2(in_channel = 3, g_conv_dim = 64, img_size = int(opt.sec_ratio))
        # decoder = Decoder1(int(opt.sec_ratio))


    if cuda:
        encoder.to(device)
        discriminator.to(device)
        decoder.cuda()

        adversarial_loss.to(device)
        cover_loss.to(device)
        if opt.deepmark_level == 2:
            perceptual_loss.to(device)
        if opt.deepmark_level == 1:
            message_loss.to(device)
        bce.to(device)

    G_losses = []
    D_losses = []

    if opt.continue_train == 0:
        # Initialize weights
        encoder.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        decoder.apply(weights_init_normal)
        # Optimizers
        optimizer_G = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    else:
        checkpoints = torch.load(DIR + "saved_models/%s/model_%d.pth"% (opt.dataset_name, opt.continue_train))
        encoder.load_state_dict(checkpoints['encoder_state_dict'])
        decoder.load_state_dict(checkpoints['decoder_state_dict'])
        discriminator.load_state_dict(checkpoints['discriminator_state_dict'])
        # Optimizers
        optimizer_G = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_G.load_state_dict(checkpoints['optimizer_G'])
        optimizer_D.load_state_dict(checkpoints['optimizer_D'])
        # Losses
        G_losses = checkpoints['g_loss_train']
        D_losses = checkpoints['d_loss_train']

    # Dataset loader
    transforms_ = [
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    if opt.phase == 'test' or opt.phase == 'extract' or opt.phase == 'embed' :
        opt.batch_size = 1
    train_set = MyImageFolder(DATA_DIR + "/train/", transforms_=transforms_)
    dataloader = DataLoader(
        train_set,
        drop_last = True,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_set = MyImageFolder(DATA_DIR + "/val/", transforms_=transforms_, mode="val", length = 500)
    val_dataloader = DataLoader(
        val_set,
        drop_last = True,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=1,
    )

    test_set = MyImageFolder(DATA_DIR + "/test/", transforms_=transforms_, mode="val", length = 50)
    test_dataloader = DataLoader(
        test_set,
        drop_last = True,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=1,
    )

    embed_set = MyImageFolder("C:/Users/maarab/Forensics/Stegano/embed/", transforms_=transforms_, mode="val")
    embed_dataloader = DataLoader(
        embed_set,
        drop_last = True,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=1,
    )


    extract_set = MyImageFolder("C:/Users/maarab/Forensics/Stegano/extracted/", transforms_=transforms_, mode="val")
    extract_dataloader = DataLoader(
        extract_set,
        drop_last = True,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=1,
    )





    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



    def bin_acc(img1, img2):
        assert img1.shape == img2.shape
        N,W,H = img1.shape
        total_acc = 0
        for iii in range(N):
            bits1 = dec2bin(img1[iii,:,:])
            bits2 = dec2bin(img2[iii,:,:])
            acc = (bits1 >= 0.0).eq(bits2 >= 0.5).sum().float().item() / bits1.numel()
            total_acc += acc
        total_acc = total_acc / N
        return total_acc

    def _acc(bin1, bin2, bit_redundancy = opt.bit_redundancy):
        assert bin1.shape == bin2.shape
        N,M = bin1.shape
        total_acc = 0
        interval = bit_redundancy
        ele_num = 8 - bit_redundancy
        
        # Custom slicing in List
        # using compress() + cycle()
        temp = cycle([True] * ele_num + [False] * interval)
        for iii in range(N):
            bits1 = bin1[iii,:]
            # print(len(bits1))
            bits1 = list(compress(bits1.tolist(), temp))
            # print(len(bits1))
            bits2 = bin2[iii,:]
            bits2 = list(compress(bits2.tolist(), temp))
            
            bits1 = np.array(bits1)
            bits2 = np.array(bits2)
            acc = np.equal((bits1 >= 0.5), (bits2 >= 0.5)).sum().astype(np.float32) / bits1.size
            # acc = (bits1 >= 0.5).eq(bits2 >= 0.5).sum().float() / bits1.numel()
            total_acc += acc
        total_acc = total_acc / N
        return total_acc

    def MBE(m1, bit_redundancy = 3):
        N,M = m1.shape
        new_m = torch.zeros((N,int(M//8)*3))
        for iii in range(N):
            x = m1[iii].chunk(opt.data_dim//8)
            for iiii in range(opt.data_dim//8):
                new_m[iii][iiii*bit_redundancy:(iiii+1)*bit_redundancy] = x[iiii][:bit_redundancy]
        return new_m
        


    def save_sample(batches_done):
        if opt.deepmark_level == 1:
            vimgs = next(iter(val_dataloader))
            vimgs, vsecret_extended, vmessage = make_pair(vimgs, opt.data_dim, use_bit_inverse=True)
            vsecret = vsecret_extended

        vconcat_imgs = torch.cat([vimgs, vsecret_extended], dim = 1)
        # print(vconcat_imgs.shape)
        if cuda:
            vimgs = vimgs.to(device)
            vsecret = vsecret.to(device)
            vsecret_extended = vsecret_extended.to(device)
            vconcat_imgs = vconcat_imgs.to(device)
            if opt.deepmark_level == 1:
                vmessage = vmessage.to(device)

        vimgs_v = Variable(vimgs)
        vsecret_v = Variable(vsecret)
        vsecret_xv = Variable(vsecret_extended)
        vconcat_imgsv = Variable(vconcat_imgs)

        vcontainer_imgs = encoder(vconcat_imgsv, vsecret_xv.float())
        noised_imgs = distortion_layers([vcontainer_imgs, vimgs_v])[0]
        if opt.deepmark_level == 1:
            
            message_shape = int(2**((math.log2(opt.data_dim)-3)//2))
            vextracted_imgs = torch.zeros(opt.batch_size, message_shape * message_shape,8, device=device)
            for jj in range(message_shape):
                for kk in range(message_shape):
                    vtest_img = noised_imgs[:,:,jj*int(opt.sec_ratio):(jj+1)*int(opt.sec_ratio),kk*int(opt.sec_ratio):(kk+1)*int(opt.sec_ratio)]
                    vextracted_imgs[:,jj*message_shape + kk,:] = decoder(vtest_img)
            vextracted_imgs = vextracted_imgs.reshape(-1,message_shape * message_shape * 8)
            # torch.set_printoptions(profile='default')
            # print("message: ", vmessage)
            # print("extracted message: ", vextracted_imgs)
            accuracy = _acc(vextracted_imgs, vmessage)

            # sample = torch.cat((vimgs.data, vcontainer_imgs.data), -2)
            save_image(vimgs.data, DIR + "images/img-%d-%f.png" % (batches_done, accuracy), nrow=opt.batch_size, normalize=True)
            save_image(vcontainer_imgs.data, DIR + "images/container_img-%d-%f.png" % (batches_done, accuracy), nrow=opt.batch_size, normalize=True)
            save_image(noised_imgs, DIR + "images/noised_img-%d-%f.png" % (batches_done, accuracy), nrow=opt.batch_size, normalize=True)

    """train"""

    # ----------
    #  Training
    # ----------
    nan_torch = torch.ones(opt.batch_size, 1)
    nan_torch[nan_torch == 1] = float('nan')
    torch.autograd.set_detect_anomaly(True)
    if opt.phase == "train":
        discriminator.train()
        start = 0
        if opt.continue_train != 0:
            start = opt.continue_train+1
        for epoch in range(start, opt.n_epochs):
        # for epoch in range(1):
            print("Training Epoch: ", epoch)
            tk0 = tqdm(dataloader, total=int(len(dataloader)))
            for i, imgs in enumerate(tk0):   
                encoder.train()
                decoder.train()
                if opt.deepmark_level == 1:
                    imgs, secret_extended, message = make_pair(imgs, opt.data_dim, use_bit_inverse=True)
                    # print(imgs.shape)
                    # print(secret_extended.shape)
                    # print(message.shape)
                    secret = secret_extended

                concat_imgs = torch.cat([imgs, secret_extended], dim = 1)
                if cuda:
                    imgs = imgs.to(device)
                    if opt.deepmark_level == 1:
                        message = message.to(device)
                    secret = secret.to(device)
                    secret_extended = secret_extended.to(device)
                    concat_imgs = concat_imgs.to(device)

                imgs_v = Variable(imgs)
                secret_v = Variable(secret)
                secret_xv = Variable(secret_extended)
                concat_imgsv = Variable(concat_imgs)

                #Adversarial ground-truths
                d_target_label_cover = torch.full((opt.batch_size,1), 1, device=device).squeeze().float()
                d_target_label_encoded = torch.full((opt.batch_size,1), 0, device=device).squeeze().float()
                g_target_label_encoded = torch.full((opt.batch_size,1), 1, device=device).squeeze().float()

                # ---------------------------
                #  Train Encoder and Decoder
                # ---------------------------
                optimizer_G.zero_grad()

                container_imgs = encoder(concat_imgsv, secret_xv.float())
                noised_imgs = distortion_layers([container_imgs, imgs_v])[0]
                if opt.deepmark_level == 1:
                    message_shape = int(2**((math.log2(opt.data_dim)-3)//2))
                    extracted_imgs = torch.zeros(opt.batch_size, message_shape * message_shape,8, device=device)
                    for j in range(message_shape):
                        for k in range(message_shape):
                            test_img = noised_imgs[:,:,j*int(opt.sec_ratio):(j+1)*int(opt.sec_ratio),k*int(opt.sec_ratio):(k+1)*int(opt.sec_ratio)]
                            # print(test_img.shape)
                            extracted_imgs[:,j*message_shape + k,:] = decoder(test_img)
                    extracted_imgs = extracted_imgs.reshape(-1,message_shape * message_shape * 8)

                #losses
                disc_outputs = discriminator(container_imgs)
                
                g_loss_adv = bce(disc_outputs, g_target_label_encoded)


                message = MBE(message)
                extracted_imgs = MBE(extracted_imgs)

                g_loss_secret = message_loss(extracted_imgs, message.float())
                g_loss_cover = cover_loss(imgs, container_imgs)
                g_loss = 0.01 * g_loss_adv + 0.09 * g_loss_secret + 0.90 * g_loss_cover

                g_loss.backward()
                optimizer_G.step()
                G_losses.append(g_loss.item())

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                d_loss_real = bce(discriminator(imgs), d_target_label_cover)
                # with torch.no_grad():
                disc_outputs = discriminator(container_imgs.detach().contiguous())
                d_loss_fake = bce(disc_outputs, d_target_label_encoded)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

                d_loss.backward()
                optimizer_D.step()

                D_losses.append(d_loss.item())

                tk0.set_postfix( g_loss_secret = g_loss_secret.item(), g_loss_cover = g_loss_cover.item(), g_loss = g_loss.item(),d_loss_fake = d_loss_fake.item(), d_loss_real = d_loss_real.item(), d_loss = d_loss.item())

                # print(
                #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, G cover: %f, G construct: %f G adv: %f G Perc: %f]"
                #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), g_cover.item(), g_construct.item(), g_adv.item(), g_perc.item())
                # )

                # Generate sample at sample interval
                batches_done = epoch * len(dataloader) + i
                if batches_done % opt.sample_interval == 0:
                    encoder.eval()
                    decoder.eval()
                    save_sample(batches_done)
            
            if epoch % 5 == 0:
                # Save model checkpoints
                save_dict = {'epoch': epoch,
                            'encoder_state_dict':encoder.state_dict(),
                            'decoder_state_dict':decoder.state_dict(),
                            'discriminator_state_dict':discriminator.state_dict(),
                            'optimizer_G': optimizer_G.state_dict(),
                            'optimizer_D': optimizer_D.state_dict(),
                            'g_loss_train': G_losses,
                            'd_loss_train': D_losses}
                torch.save(save_dict, DIR + "saved_models/%s/model_%d.pth"% (opt.dataset_name, epoch))

    """test - level 1"""



    def computePSNR(origin,pred):
        normalize = np.amax(origin)-np.amin(origin)

        origin = np.array(origin)
        origin = origin.astype(np.float32)
        origin = (origin/normalize)*255
        pred = np.array(pred)
        pred = pred.astype(np.float32)
        pred = (pred/normalize)*255

        mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
        if mse < 1.0e-10:
            return 10 * math.log10(255.0**2/mse)

    def calculate_psnr(img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


    def calculate_ssim(img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')

        # img1 = np.array(oriimg1gin)
        # img1 = img1.astype(np.float32)
        # img2 = np.array(img2)
        # img2 = img2.astype(np.float32)
        img1 = np.transpose(img1)
        img1 = (img1/2)*255
        img2 = np.transpose(img2)
        img2 = (img2/2)*255
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:       
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    temp_ssim = ssim(img1, img2)
                    ssims.append(temp_ssim)
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')



    def convert_to_comp(imgs, comp_type = "Compression"):
            comp_imgs = torch.zeros((imgs.shape))
            
            for i in range(imgs.shape[0]):
                imgs[i] = imgs[i]*0.5+0.5
                pil_img = transforms.ToPILImage()(imgs[i])
                # pil_img = imgs[i]

                
                if comp_type == "WebP":
                    pil_img.save(str(i+1) + ".webp", format="webp")
                    comp_img = Image.open(str(i+1) + ".webp").convert('RGB')
                elif comp_type == "JPEG2000":
                    pil_img.save(str(i+1) + ".jp2", format = "JPEG2000")
                    comp_img = Image.open(str(i+1)+".jp2").convert('RGB')
                elif comp_type == "Compression":
                    pil_img.save(str(i+1) + ".jpg", format = "JPEG", quality=80)
                    comp_img = Image.open(str(i+1)+".jpg").convert('RGB')
                    # pil_img.save(str(i+1) + ".png", format = "PNG")
                    # comp_img = Image.open(str(i+1)+".png").convert('RGB')

                comp_img = transforms.PILToTensor()(comp_img).float()
                comp_img = transforms.CenterCrop(opt.img_size)(comp_img)
                comp_img = comp_img/255.0
                comp_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(comp_img)
                
                comp_imgs[i] = comp_img
            return comp_imgs

    lines = []
    if opt.phase == "test":

        calc_SSIM_PSNR = True
        if opt.use_noise == False or noises == []:
            noises = ["No noise"]
        # noises=["Compression"]
        file_path = "messages.txt"
        for n, noise_str in enumerate(noises):
            # if noise_str == "Compression":
            #     continue
            psnr_s1 = []
            psnr_s2 = []
            ssim_s = []
            encoder.eval()
            decoder.eval()
            lines = ["testing for the following distortions:"]
            lines.append(noise_str)

            print("testing for the following distortions:")
            # print(list(noise_str))
            if opt.use_noise:
                noise = noises[n:n+1]
                print(noise)
            else:
                noise = []
            distortion_layers = DistortionLayer(noise, infoDict, device, "test")
            distortion_layers.to(device)
            # for ijk in range(0,8):
            for bit_redundancy in range(7,8):
                print("testing for bit redundancy of(",str(bit_redundancy),"/ 7)...")
                tk1 = tqdm(test_dataloader, total=int(len(test_dataloader)))
                count = 0
                accuracy_total = 0.0
                if calc_SSIM_PSNR:
                    psnr_total = 0.0
                    ssim_total = 0.0
                for i, vimgs in enumerate(tk1):
                    
                    vimgs, vsecret_extended, vmessage = make_pair(vimgs, opt.data_dim, use_bit_inverse=True)
                    vsecret = vsecret_extended

                    # print(vmessage.numpy())
                    # with open(file_path, 'a') as file:
                    #     # Convert the tensor to a string representation and write it to the file
                    #     tensor_str = ','.join(map(str, vmessage.view(-1).tolist()))
                    #     file.write(tensor_str + '\n')

                    vconcat_imgs = torch.cat([vimgs, vsecret_extended], dim = 1)
                    # print(vconcat_imgs.shape)
                    if cuda:
                        vimgs = vimgs.to(device)
                        vsecret = vsecret.to(device)
                        vsecret_extended = vsecret_extended.to(device)
                        vconcat_imgs = vconcat_imgs.to(device)
                        vmessage = vmessage.to(device)

                    vimgs_v = Variable(vimgs)
                    vsecret_v = Variable(vsecret)
                    vsecret_xv = Variable(vsecret_extended)
                    vconcat_imgsv = Variable(vconcat_imgs)

                    vcontainer_imgs = encoder(vconcat_imgsv, vsecret_xv.float())
                    if noise == "Compression" or noise == "JPEG2000" or noise == "WebP":
                        noised_imgs = convert_to_comp(vcontainer_imgs, comp_type = noise).to(device)
                    else:
                        noised_imgs = distortion_layers([vcontainer_imgs, vimgs_v])[0]
                        
                    message_shape = int(2**((math.log2(opt.data_dim)-3)//2))
                    vextracted_imgs = torch.zeros(opt.batch_size, message_shape * message_shape,8, device=device)
                    for jj in range(message_shape):
                        for kk in range(message_shape):
                            vtest_img = noised_imgs[:,:,jj*int(opt.sec_ratio):(jj+1)*int(opt.sec_ratio),kk*int(opt.sec_ratio):(kk+1)*int(opt.sec_ratio)]
                            vextracted_imgs[:,jj*message_shape + kk,:] = decoder(vtest_img)
                    vextracted_imgs = vextracted_imgs.reshape(-1,message_shape * message_shape * 8)

                    # torch.set_printoptions(threshold=10000)
                    # x = vextracted_imgs
                    # x = (x>=0.5)
                    # x = x.type(torch.uint8)
                    # print("message: ", torch.reshape(vmessage, (-1,)))
                    # print("extracted message: ", torch.reshape(x, (-1,)))

                    accuracy = _acc(vextracted_imgs, vmessage, bit_redundancy)
                    accuracy_total += accuracy
                    
                    if bit_redundancy == 7 and calc_SSIM_PSNR:
                        # cover = vcontainer_imgs[0].cpu().detach().numpy().squeeze()
                        # container = torch.squeeze(vimgs).cpu().numpy().squeeze()
                        # noised = noised_imgs[0].cpu().detach().numpy().squeeze()
                        # print("cover: ", cover)
                        # print("container: ", container)
                        # print("difference: ", cover-container)
                        psnr_temp1 = peak_signal_noise_ratio(vcontainer_imgs.detach().cpu().numpy(), vimgs.detach().cpu().numpy())
                        # psnr_temp2 = calculate_psnr(cover, container)
                        psnr_s1.append(psnr_temp1)
                        # psnr_s2.append(psnr_temp2)
                        # ssim_temp = structural_similarity(vcontainer_imgs.detach().cpu().numpy(), vimgs.detach().cpu().numpy(), multichannel=True)
                        # ssim_s.append(ssim_temp)
                    if calc_SSIM_PSNR:
                        tk1.set_postfix(accuracy = accuracy, PSNR1 = psnr_temp1)#, SSIM = ssim_temp)
                    else:
                        tk1.set_postfix(accuracy = accuracy)
                    save_image(vimgs.data, DIR + "test_images/img_"+noise_str+"-%d.png" % (count), nrow=opt.batch_size, normalize=True)
                    save_image(vcontainer_imgs.data, DIR + "test_images/container_img_"+noise_str+"-%d.png" % (count), nrow=opt.batch_size, normalize=True)
                    # save_image(noised_imgs.data, DIR + "test_images/noised_img_"+noise_str+"-%d.png" % (count), nrow=opt.batch_size, normalize=True)
                    count += 1

                accuracy = accuracy_total/count
                lines.append("The accuracy for " + str(int(((8 - bit_redundancy)/8)*opt.data_dim)) + "bits of secret message with reduncancy of " + str(int((bit_redundancy/8)*opt.data_dim)) + " bits is: " + str(accuracy)+ "\%")
                if bit_redundancy == 6 and calc_SSIM_PSNR:
                    lines.append("The average PSNR is: "+ str(np.mean(psnr_s1)))
                    # lines.append("The average SSIM is: "+ str(np.mean(ssim_s)))
                print("The accuracy for " + str(int(((8 - bit_redundancy)/8)*opt.data_dim)) + "bits of secret message with reduncancy of " + str(int((bit_redundancy/8)*opt.data_dim)) + " bits is: " + str(accuracy)+ "\%")
                if calc_SSIM_PSNR:
                    print("PSNR: ", np.mean(psnr_s1))
                    # print("PSNR2: ", np.mean(psnr_s2))
                    # print("SSIM: ", np.mean(ssim_s))
            with open(DIR + 'test_results'+opt.dataset_name +'.txt', 'a') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

    if opt.phase == 'embed':
        calc_SSIM_PSNR = False
        encoder.eval()
        decoder.eval()
        # for ijk in range(0,8):
        tk1 = tqdm(embed_dataloader, total=int(len(embed_dataloader)))
        count = 0
        accuracy_total = 0.0
        if calc_SSIM_PSNR:
            psnr_total = 0.0
            ssim_total = 0.0
        # for i in range(len(embed_set)):
        for i, vimgs in enumerate(tk1):
            # vimgs = next(iter(embed_dataloader))
            count = count + 1
            
            vimgs, vsecret_extended, vmessage = make_pair(vimgs, opt.data_dim, use_bit_inverse=True)
            vsecret = vsecret_extended

            vconcat_imgs = torch.cat([vimgs, vsecret_extended], dim = 1)
            # print(vconcat_imgs.shape)
            if cuda:
                vimgs = vimgs.to(device)
                vsecret = vsecret.to(device)
                vsecret_extended = vsecret_extended.to(device)
                vconcat_imgs = vconcat_imgs.to(device)
                vmessage = vmessage.to(device)

            vimgs_v = Variable(vimgs)
            vsecret_v = Variable(vsecret)
            vsecret_xv = Variable(vsecret_extended)
            vconcat_imgsv = Variable(vconcat_imgs)

            vcontainer_imgs = encoder(vconcat_imgsv, vsecret_xv.float())

            save_image(vcontainer_imgs.data, "C:/Users/maarab/Forensics/Stegano/embedded/%d.png" % (count), nrow=opt.batch_size, normalize=True)
        
            
            embedded = vmessage.to(torch.int)
            print(embedded.view(-1).tolist())
            with open("./embedded_message.txt", 'a') as file:
                # Convert the tensor to a string representation and write it to the file
                tensor_str = ','.join(map(str, embedded.view(-1).tolist()))
                file.write(tensor_str + '\n')


    if opt.phase == 'extract':
        calc_SSIM_PSNR = False
        encoder.eval()
        decoder.eval()
        # for ijk in range(0,8):
        tk1 = tqdm(extract_dataloader, total=int(len(extract_dataloader)))
        count = 0
        accuracy_total = 0.0
        if calc_SSIM_PSNR:
            psnr_total = 0.0
            ssim_total = 0.0
        for i, vimgs in enumerate(tk1):
            
            vimgs, vsecret_extended, vmessage = make_pair(vimgs, opt.data_dim, use_bit_inverse=True)
            vsecret = vsecret_extended

            # print(vconcat_imgs.shape)
            if cuda:
                vimgs = vimgs.to(device)
                vsecret = vsecret.to(device)
                vsecret_extended = vsecret_extended.to(device)

                vmessage = vmessage.to(device)

            vimgs_v = Variable(vimgs)
            vsecret_v = Variable(vsecret)
            vsecret_xv = Variable(vsecret_extended)

   
            message_shape = int(2**((math.log2(opt.data_dim)-3)//2))
            vextracted_imgs = torch.zeros(opt.batch_size, message_shape * message_shape,8, device=device)
            for jj in range(message_shape):
                for kk in range(message_shape):
                    vtest_img = vimgs_v[:,:,jj*int(opt.sec_ratio):(jj+1)*int(opt.sec_ratio),kk*int(opt.sec_ratio):(kk+1)*int(opt.sec_ratio)]
                    vextracted_imgs[:,jj*message_shape + kk,:] = decoder(vtest_img)
            vextracted_imgs = vextracted_imgs.reshape(-1,message_shape * message_shape * 8)

            # print(vextracted_imgs.shape)

            # torch.set_printoptions(threshold=10000)
            # x = vextracted_imgs
            # x = (x>=0.5)
            # x = x.type(torch.uint8)
            # print("message: ", torch.reshape(vmessage, (-1,)))
            # print("extracted message: ", torch.reshape(x, (-1,)))
            extracted = torch.round(vextracted_imgs).to(torch.int)
            print(extracted.view(-1).tolist())
            with open("./extracted_message.txt", 'a') as file:
                # Convert the tensor to a string representation and write it to the file
                tensor_str = ','.join(map(str, extracted.view(-1).tolist()))
                file.write(tensor_str + '\n')

if __name__ == "__main__":
    main()