import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride of the convolution layer

        return: residual block
        """
        super(Block, self).__init__()

        self.conv1_block = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=3 // 2
        )
        self.bn1_block = nn.BatchNorm2d(num_features=out_channels)

        self.relu_block = nn.ReLU(inplace=True)

        self.conv2_block = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=3 // 2
        )

        self.bn2_block = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = self.conv1_block(x)
        out = self.bn1_block(out)
        out = self.relu_block(out)
        out = self.conv2_block(out)
        out = self.bn2_block(out)
        out = out + x
        out = self.relu_block(out)
        return out


class ResNet_18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_18, self).__init__()
        """
        ResNet-18
        """
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=7 // 2
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        depth = [2, 2, 2, 2]

        layer = []
        for i in range(depth[0]):
            if i == 0:
                layer.append(Block(64, 64, 2))
            else:
                layer.append(Block(64, 64, 1))
        self.layer1 = nn.Sequential(*layer)

        layer = []
        for i in range(depth[1]):
            if i == 0:
                layer.append(Block(64, 128, 2))
            else:
                layer.append(Block(128, 128, 1))
        self.layer2 = nn.Sequential(*layer)

        layer = []
        for i in range(depth[2]):
            if i == 0:
                layer.append(Block(128, 256, 2))
            else:
                layer.append(Block(256, 256, 1))
        self.layer3 = nn.Sequential(*layer)

        layer = []
        for i in range(depth[3]):
            if i == 0:
                layer.append(Block(256, 512, 2))
            else:
                layer.append(Block(512, 512, 1))
        self.layer4 = nn.Sequential(*layer)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

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
