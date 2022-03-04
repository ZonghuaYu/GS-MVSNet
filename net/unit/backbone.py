import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FPN_2layers(nn.Module):
    def __init__(self, inner_channels=[3, 8, 16, 32], out_channels=16):
        super(FPN_2layers, self).__init__()

        in0, in1, in2, in3 = inner_channels
        # [B,8,H,W]
        self.conv0 = nn.Sequential(
            ConvBnReLU(in0, in1, 3, 1, 1),
            ConvBnReLU(in1, in1, 3, 1, 1),
            ConvBnReLU(in1, in2, 3, 1, 1),
            ConvBnReLU(in2, in2, 3, 1, 1),
            ConvBnReLU(in2, in2, 3, 1, 1),
            ConvBnReLU(in2, in2, 3, 1, 1)
        )

        # [B,16,H/2,W/2]
        self.conv1 = nn.Sequential(
            ConvBnReLU(in2, in3, 3, 1, 1),
            ConvBnReLU(in3, in3, 3, 2, 1),
            ConvBnReLU(in3, in3, 3, 1, 1),
            ConvBnReLU(in3, in3, 3, 1, 1)
        )

        self.lat = nn.Conv2d(in2, in3, 1)
        self.fuse = nn.Conv2d(in3, out_channels, 1, bias=False)

    def forward(self, x):

        x = self.conv0(x)
        x = self.lat(x) + F.interpolate(self.conv1(x), scale_factor=2.0, mode="bilinear", align_corners=False)

        return self.fuse(x)


if __name__=="__main__":
    x = torch.randint(1,255,(1,3,128,160)).float()
    net =  FPN_2layers()
    print('>>>Number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    outs = net(x)

    for y in outs:
        print(y.shape)
