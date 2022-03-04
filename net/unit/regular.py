import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RegularNet(nn.Module):
    def __init__(self, in_channels=16, inner_channels=[16, 32, 64]):
        super(RegularNet, self).__init__()
        c0, c1, c2 = inner_channels

        self.conv0 = nn.Sequential(
            ConvBnReLU3D(in_channels, c0, kernel_size=3, padding=1),
            ConvBnReLU3D(c0, c0, kernel_size=3, padding=1),
        )

        self.conv1 = nn.Sequential(
            ConvBnReLU3D(c0, c1, kernel_size=3, stride=2, padding=1),
            ConvBnReLU3D(c1, c1, kernel_size=3, padding=1),
            ConvBnReLU3D(c1, c1, kernel_size=3, padding=1),
            ConvBnReLU3D(c1, c2, kernel_size=3, padding=1),
            ConvBnReLU3D(c2, c2, kernel_size=3, padding=1),
            ConvBnReLU3D(c2, c2, kernel_size=3, padding=1),
        )

        self.conv20 = nn.Sequential(
            nn.ConvTranspose3d(c2, c1, kernel_size=3, padding=1, output_padding=0, stride=1, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(c1, c0, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(c0),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(c0, 1, 3, stride=1, padding=1)

    def forward(self, x):
        """
        3D CNN regular
        @param x: cost volume:(B,C,D,H,W)
        @return: prob volume；(B,D,H,W)
        """
        H, W = x.shape[-2:]
        assert (H%2==0 and W%2==0), "the shape of input image must can div 2!"+"current cost volume shape:"+str(x.shape)

        x = self.conv0(x)
        x = x + self.conv20(self.conv1(x))
        x = self.prob(x).squeeze(1)

        return F.softmax(x, dim=1)


if __name__=="__main__":
    r =RegularNet()

    import torch
    x = torch.randn(2,16,8,100,100)
    y = r(x)
    print(y.shape)

