import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, kernel_size=1, dw_size=3, stride=1):
        super().__init__()

        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size,
                      stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.primary_conv(x)
        z = self.cheap_conv(y)
        return torch.cat([y, z], dim=1)

class GhostResBlock(nn.Module):
    """使用Ghost模块的残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 判断是否需要下采样
        self.downsample = (stride != 1) or (in_channels != out_channels)

        # 主路径
        self.conv1 = GhostModule(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, groups=out_channels, bias=False),  # 深度卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = GhostModule(out_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 捷径
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = F.relu(out)

        return out

class MicroResNetGhost(nn.Module):
    """使用Ghost模块的轻量级ResNet"""

    def __init__(self, num_classes=100):
        super().__init__()

        # 第一个卷积层 - 保持原样，因为输入通道数较少
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 使用Ghost残差块
        self.block1 = self._make_ghost_block(16, 16, stride=1)
        self.block2 = self._make_ghost_block(16, 32, stride=2)
        self.block3 = self._make_ghost_block(32, 64, stride=2)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_ghost_block(self, in_channels, out_channels, stride):
        """创建Ghost残差块"""
        return GhostResBlock(in_channels, out_channels, stride)

    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        x = self.fc(feature)

        if return_features:
            return x, feature
        return x