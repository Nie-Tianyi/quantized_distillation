import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# --------------------------------------------------
#             Squeeze-Excitation 注意力模块
# --------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.squeeze(x)
        w = self.excite(w)
        return x * w


# --------------------------------------------------
#        Depthwise + Pointwise 卷积 (DW + PW)
# --------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.se(x)
        return x


# --------------------------------------------------
#        Residual block (DWConv + SE + shortcut)
# --------------------------------------------------
class DWResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv = DepthwiseSeparableConv(in_ch, out_ch, stride)

        # 是否需要映射 shortcut
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# --------------------------------------------------
#                     TinyCNN++
# --------------------------------------------------
class TinyCNNPlusPlus(nn.Module):
    """
    TinyCNN++:
    - Depthwise Separable Conv
    - SE Attention
    - Residual Blocks
    - Global Average Pooling
    """
    def __init__(self, num_classes=100):
        super().__init__()

        self.stage1 = DWResidualBlock(3, 32, stride=1)
        self.stage2 = DWResidualBlock(32, 64, stride=2)
        self.stage3 = DWResidualBlock(64, 128, stride=2)
        self.stage4 = DWResidualBlock(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        features = self.gap(x).flatten(1)
        logits = self.fc(features)

        return (logits, features) if return_features else logits



# --------------------------------------------------
#             Squeeze-Excitation 注意力模块
# --------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.squeeze(x)
        w = self.excite(w)
        return x * w


# --------------------------------------------------
#        Depthwise + Pointwise 卷积 (DW + PW)
# --------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        # Depthwise
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False
        )

        # Pointwise
        self.pw = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, bias=False
        )

        self.bn = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.se(x)
        return x

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, kernel_size=1, dw_size=3, stride=1):
        super().__init__()

        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size,
                      stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.primary_conv(x)
        z = self.cheap_conv(y)
        return torch.cat([y, z], dim=1)

def channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batchsize, -1, height, width)


class GhostShuffleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        mid = out_ch // 2

        # 1. Ghost pointwise conv (expand)
        self.ghost1 = GhostModule(in_ch, mid, kernel_size=1)

        # 2. Depthwise conv
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # 3. Ghost pointwise conv (project)
        self.ghost2 = GhostModule(mid, out_ch, kernel_size=1)

        # 4. Shortcut
        self.shortcut = (nn.Sequential()
            if stride == 1 and in_ch == out_ch else
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        )

    def forward(self, x):
        out = self.ghost1(x)
        out = self.dw(out)
        out = self.ghost2(out)

        out = channel_shuffle(out, groups=2)

        return out + self.shortcut(x)

class TinyCNNGhostShuffle(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.stage1 = GhostShuffleBlock(3, 32, stride=1)
        self.stage2 = GhostShuffleBlock(32, 64, stride=2)
        self.stage3 = GhostShuffleBlock(64, 128, stride=2)
        self.stage4 = GhostShuffleBlock(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        feat = self.gap(x).flatten(1)
        logits = self.fc(feat)

        return (logits, feat) if return_features else logits
