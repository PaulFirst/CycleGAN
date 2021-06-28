import torch.nn as nn
from src.model.blocks import *


class Generator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Generator, self).__init__()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=(7, 7), padding=3),
                                         nn.InstanceNorm2d(64), nn.ReLU())  # 64x256x256
        self.conv_block2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                         nn.InstanceNorm2d(128), nn.ReLU())  # 128x128x128
        self.conv_block3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                         nn.InstanceNorm2d(256), nn.ReLU())  # 256x64x64

        self.res1 = ResidualBlock(256, kernel=(3, 3), stride=(1, 1), padding=1)
        self.res2 = ResidualBlock(256, kernel=(3, 3), stride=(1, 1), padding=1)
        self.res3 = ResidualBlock(256, kernel=(3, 3), stride=(1, 1), padding=1)
        self.res4 = ResidualBlock(256, kernel=(3, 3), stride=(1, 1), padding=1)
        self.res5 = ResidualBlock(256, kernel=(3, 3), stride=(1, 1), padding=1)
        self.res6 = ResidualBlock(256, kernel=(3, 3), stride=(1, 1), padding=1)
        self.res7 = ResidualBlock(256, kernel=(3, 3), stride=(1, 1), padding=1)
        self.res8 = ResidualBlock(256, kernel=(3, 3), stride=(1, 1), padding=1)
        self.res9 = ResidualBlock(256, kernel=(3, 3), stride=(1, 1), padding=1)

        self.upsample_block1 = nn.Sequential(Upsample(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                             nn.InstanceNorm2d(128), nn.ReLU())  # 128x128x128
        self.upsample_block2 = nn.Sequential(Upsample(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                             nn.InstanceNorm2d(64), nn.ReLU())  # 64x256x256

        self.conv_block4 = nn.Sequential(nn.Conv2d(64, out_ch, kernel_size=(7, 7), padding=3),
                                         nn.Tanh())  # outx256x256

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.upsample_block1(x)
        x = self.upsample_block2(x)
        out = self.conv_block4(x)
        return out
