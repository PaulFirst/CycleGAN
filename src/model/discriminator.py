import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_ch):
        super(Discriminator, self).__init__()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                         nn.LeakyReLU(0.2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.LeakyReLU(0.2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                         nn.InstanceNorm2d(256),
                                         nn.LeakyReLU(0.2))
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4, 4), padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2))
        self.conv_block5 = nn.Conv2d(512, 1, kernel_size=(4, 4), padding=1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        out = self.conv_block5(x)
        return out
