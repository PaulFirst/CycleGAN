import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel, stride, padding):
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(ch, ch, kernel, stride, padding), nn.InstanceNorm2d(ch),
                                 nn.ReLU(),
                                 nn.Conv2d(ch, ch, kernel, stride, padding), nn.InstanceNorm2d(ch))

    def forward(self, x):
        return x + self.res(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.refl = nn.ReflectionPad2d(1),
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, inp):
        inp = self.ups(inp)
        #inp = self.refl(inp)
        return self.conv(inp)
