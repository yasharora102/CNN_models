# EfficientNet-b0 from scrratch in Pytorchx
import torch
import torch.nn as nn


class Conv_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        bias=False,
        use_bn=True,
        use_act=True,
    ):
        super(Conv_Block, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.act = nn.SiLU(inplace=True) if use_act else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x) if self.act is not None else x
        return x


# squeeze and excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SEBlock, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels, reduced_dim, kernel_size=1, stride=1, padding=0
        )
        self.fc2 = nn.Conv2d(
            reduced_dim, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.act = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.global_avg_pool(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


# MBConv block
class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        reduction=16,
    ):
        super(MBConv, self).__init__()
        self.stride = stride
        self.skip = True if self.stride == 1 and in_channels == out_channels else False
        self.expand = in_channels * expand_ratio
        reduced_dim = in_channels // reduction
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            self.conv1 = Conv_Block(
                in_channels,
                self.expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        else:
            self.expand_conv = None
        # depthwise conv
        self.conv2 = Conv_Block(
            self.expand,
            self.expand,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=self.expand,
            bias=False,
        )

        # squeeze and excitation
        self.se = SEBlock(self.expand, reduced_dim)

        # project conv
        self.conv3 = Conv_Block(
            self.expand,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            use_act=False,
        )

    def forward(self, x):
        y = self.conv1(x) if self.expand_ratio != 1 else x
        y = self.conv2(y)
        y = self.se(y)
        y = self.conv3(y)

        if self.skip:
            y = x + y

        return y


class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNet_b0, self).__init__()

        self.conv1 = Conv_Block(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.layer1 = MBConv(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            expand_ratio=1,
        )
        # depth 2
        self.layer2 = nn.Sequential(
            MBConv(
                in_channels=16,
                out_channels=24,
                kernel_size=3,
                stride=2,
                expand_ratio=6,
            ),
            MBConv(
                in_channels=24,
                out_channels=24,
                kernel_size=3,
                stride=1,
                expand_ratio=6,
            ),
        )

        # depth 2
        self.layer3 = nn.Sequential(
            MBConv(
                in_channels=24,
                out_channels=40,
                kernel_size=5,
                stride=2,
                expand_ratio=6,
            ),
            MBConv(
                in_channels=40,
                out_channels=40,
                kernel_size=5,
                stride=1,
                expand_ratio=6,
            ),
        )

        # depth 3
        self.layer4 = nn.Sequential(
            MBConv(
                in_channels=40,
                out_channels=80,
                kernel_size=3,
                stride=2,
                expand_ratio=6,
            ),
            MBConv(
                in_channels=80,
                out_channels=80,
                kernel_size=3,
                stride=1,
                expand_ratio=6,
            ),
            MBConv(
                in_channels=80,
                out_channels=80,
                kernel_size=3,
                stride=1,
                expand_ratio=6,
            ),
        )
        # depth 3
        self.layer5 = nn.Sequential(
            MBConv(
                in_channels=80,
                out_channels=112,
                kernel_size=5,
                stride=1,
                expand_ratio=6,
            ),
            MBConv(
                in_channels=112,
                out_channels=112,
                kernel_size=5,
                stride=1,
                expand_ratio=6,
            ),
            MBConv(
                in_channels=112,
                out_channels=112,
                kernel_size=5,
                stride=1,
                expand_ratio=6,
            ),
        )
        # depth 4
        self.layer6 = nn.Sequential(
            MBConv(
                in_channels=112,
                out_channels=192,
                kernel_size=5,
                stride=2,
                expand_ratio=6,
            ),
            MBConv(
                in_channels=192,
                out_channels=192,
                kernel_size=5,
                stride=1,
                expand_ratio=6,
            ),
            MBConv(
                in_channels=192,
                out_channels=192,
                kernel_size=5,
                stride=1,
                expand_ratio=6,
            ),
            MBConv(
                in_channels=192,
                out_channels=192,
                kernel_size=5,
                stride=1,
                expand_ratio=6,
            ),
        )
        self.layer7 = MBConv(
            in_channels=192,
            out_channels=320,
            kernel_size=3,
            stride=1,
            expand_ratio=6,
        )

        self.conv2 = Conv_Block(
            in_channels=320,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
