import torch
import torch.nn as nn


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """
    args:
        in_channels: number of input channels
        out_channels: number of output channels
        stride: stride of the convolution layer
        padding: padding of the convolution layer
    """

    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
    )


class Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, downsample=False, down_stride=0
    ):
        """
        args:

            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride of the convolution layer
            downsample: if True, downsample the input
            down_stride: stride of the downsample convolution layer
        """
        super(Block, self).__init__()

        self.conv1_block = conv1x1(in_channels, out_channels, stride=1, padding=0)
        self.bn1_block = nn.BatchNorm2d(num_features=out_channels)

        if downsample:
            self.conv2_block = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=down_stride,
                padding=3 // 2,
                groups=32,
                bias=False,
            )
        else:
            self.conv2_block = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
                groups=32,
                bias=False,
            )
        self.bn2_block = nn.BatchNorm2d(num_features=out_channels)
        self.relu_block = nn.ReLU(inplace=True)
        self.conv3_block = conv1x1(out_channels, out_channels * 2, stride=1, padding=0)
        self.bn3_block = nn.BatchNorm2d(num_features=out_channels * 2)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 2,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(num_features=out_channels * 2),
            )

        else:
            self.downsample = None

    def forward(self, x):
        out = self.conv1_block(x)
        out = self.bn1_block(out)
        out = self.relu_block(out)
        out = self.conv2_block(out)
        out = self.bn2_block(out)
        out = self.relu_block(out)
        out = self.conv3_block(out)
        out = self.bn3_block(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x
        out = self.relu_block(out)
        return out


class ResNeXt_50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNeXt_50, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=7 // 2,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        depth = [3, 4, 6, 3]

        layer = []
        for i in range(depth[0]):
            if i == 0:
                layer.append(Block(64, 128, 1, downsample=True, down_stride=1))

            else:
                layer.append(Block(256, 128, 1))

        self.layer1 = nn.Sequential(*layer)

        layer = []
        for i in range(depth[1]):
            if i == 0:
                layer.append(Block(256, 256, 2, downsample=True, down_stride=2))
            else:
                layer.append(Block(512, 256, 1))
        self.layer2 = nn.Sequential(*layer)

        layer = []
        for i in range(depth[2]):
            if i == 0:
                layer.append(Block(512, 512, 2, downsample=True, down_stride=2))
            else:
                layer.append(Block(1024, 512, 1))
        self.layer3 = nn.Sequential(*layer)

        layer = []
        for i in range(depth[3]):
            if i == 0:
                layer.append(Block(1024, 1024, 2, downsample=True, down_stride=2))
            else:
                layer.append(Block(2048, 1024, 1))
        self.layer4 = nn.Sequential(*layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
