import torch
import torch.nn as nn


class Aux_Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        args:
            in_channels: number of input channels
            num_classes: number of output classes
        """
        super(Aux_Classifier, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, features):
        """
        args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride of the convolution layer
            features: list of features
        """
        super(Inception_Block, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=features[0]),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=features[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features[1],
                out_channels=features[2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=features[2]),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=features[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[4],
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(num_features=features[4]),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(
                in_channels,
                out_channels=features[5],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=features[5]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)

        output = torch.cat([x1, x2, x3, x4], dim=1)
        return output


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()

        features = [
            [64, 96, 128, 16, 32, 32],  # 3a
            [128, 128, 192, 32, 96, 64],  # 3b
            [192, 96, 208, 16, 48, 64],  # 4a
            [160, 112, 224, 24, 64, 64],  # 4b
            [128, 128, 256, 24, 64, 64],  # 4c
            [112, 144, 288, 32, 64, 64],  # 4d
            [256, 160, 320, 32, 128, 128],  # 4e
            [256, 160, 320, 32, 128, 128],  # 5a
            [384, 192, 384, 48, 128, 128],  # 5b
        ]

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.inception3a = Inception_Block(
            in_channels=192, out_channels=256, stride=1, features=features[0]
        )
        self.inception3b = Inception_Block(
            in_channels=256, out_channels=480, stride=1, features=features[1]
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception_Block(
            in_channels=480, out_channels=512, stride=1, features=features[2]
        )
        self.inception4b = Inception_Block(
            in_channels=512, out_channels=512, stride=1, features=features[3]
        )
        self.aux1 = Aux_Classifier(512, num_classes)
        self.aux2 = Aux_Classifier(528, num_classes)

        self.inception4c = Inception_Block(
            in_channels=512, out_channels=512, stride=1, features=features[4]
        )
        self.inception4d = Inception_Block(
            in_channels=512, out_channels=528, stride=1, features=features[5]
        )
        self.inception4e = Inception_Block(
            in_channels=528, out_channels=832, stride=1, features=features[6]
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_Block(
            in_channels=832, out_channels=832, stride=1, features=features[7]
        )
        self.inception5b = Inception_Block(
            in_channels=832, out_channels=1024, stride=1, features=features[8]
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool1(x)
        x = self.inception4a(x)
        aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool2(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return aux1, aux2, x
