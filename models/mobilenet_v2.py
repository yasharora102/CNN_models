import torch
import torch.nn as nn


class Inv_Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, features):
        """
        args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride of the convolution layer
            features: list of features
        """
        super(Inv_Residual, self).__init__()

        expansion = features[0]
        self.stride = stride

        # 1x1 conv
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=in_channels * expansion),
            nn.ReLU6(inplace=True),
            # 3x3 conv depthwise
            nn.Conv2d(
                in_channels * expansion,
                in_channels * expansion,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * expansion,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU6(inplace=True),
            # 1x1 conv
            nn.Conv2d(
                in_channels * expansion,
                features[1],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features[1]),
        )
        self.skip = True if self.stride == 1 and in_channels == out_channels else False

    def forward(self, x):
        if self.skip:
            out = self.conv(x) + x
        else:
            out = self.conv(x)
        return out


class Mobilenet_V2(nn.Module):
    def __init__(self, num_classes=10):
        super(Mobilenet_V2, self).__init__()

        features = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU6(inplace=True),
        )

        self.layer1 = Inv_Residual(32, 16, 1, features[0])

        self.layer2 = nn.Sequential(
            Inv_Residual(16, 24, 2, features[1]),
            Inv_Residual(24, 24, 1, features[1]),
        )

        self.layer3 = nn.Sequential(
            Inv_Residual(24, 32, 2, features[2]),
            Inv_Residual(32, 32, 1, features[2]),
            Inv_Residual(32, 32, 1, features[2]),
        )

        self.layer4 = nn.Sequential(
            Inv_Residual(32, 64, 2, features[3]),
            Inv_Residual(64, 64, 1, features[3]),
            Inv_Residual(64, 64, 1, features[3]),
            Inv_Residual(64, 64, 1, features[3]),
        )

        self.layer5 = nn.Sequential(
            Inv_Residual(64, 96, 1, features[4]),
            Inv_Residual(96, 96, 1, features[4]),
            Inv_Residual(96, 96, 1, features[4]),
        )

        self.layer6 = nn.Sequential(
            Inv_Residual(96, 160, 2, features[5]),
            Inv_Residual(160, 160, 1, features[5]),
            Inv_Residual(160, 160, 1, features[5]),
        )

        self.layer7 = Inv_Residual(160, 320, 1, features[6])

        self.layer8 = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0),
            nn.BatchNorm2d(num_features=1280),
            nn.ReLU6(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
