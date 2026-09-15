import torch
import torch.nn as nn

class CoordinateAttention(nn.Module):
    def __init__(self, c1, c2, ratio=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid = max(8, c1 // ratio)
        self.conv1 = nn.Conv2d(c1, mid, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mid, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid, c2, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        return x * self.sigmoid(self.conv_h(x_h)) * self.sigmoid(self.conv_w(x_w))