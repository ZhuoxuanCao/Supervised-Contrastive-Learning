import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, BatchNorm2d, Dropout


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res=True):
        super(ConvBlock, self).__init__()
        self.res = res
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if res and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResModel(nn.Module):
    def __init__(self, res=True):
        super(ResModel, self).__init__()
        self.block1 = ConvBlock(3, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.block4 = ConvBlock(256, 512)
        self.classifier = nn.Sequential(Flatten(),  # 7 Flatten 层
                                        Dropout(0.4),
                                        Linear(2048, 256),  # 8 全连接层
                                        Linear(256, 64),  # 8 全连接层
                                        Linear(64, 10))  # 9 全连接层 ) # fc，最终 Cifar10输出是10类
        self.relu = ReLU(inplace=True)
        self.maxpool = Sequential(MaxPool2d(kernel_size=2))  # 1 最大池化层

    def forward(self, x):
        out = self.block1(x)
        out = self.maxpool(out)
        # print(out.shape)  # 打印第一个池化层后的输出形状
        out = self.block2(out)
        out = self.maxpool(out)
        # print(out.shape)  # 打印第三个池化层后的输出形状
        out = self.block3(out)
        out = self.maxpool(out)
        # print(out.shape)  # 打印第三个池化层后的输出形状
        out = self.block4(out)
        out = self.maxpool(out)  # 此时 out 的尺寸是 torch.Size([50, 512, 2, 2])
        # print(out.shape)  # 打印第四个池化层后的输出形状

        # out = torch.flatten(out, start_dim=1)  # 展平从第二维开始，即展平 512*2*2 -> 50*2048
        # 或者使用 nn.Flatten()，如果使用这个，则无需指定 start_dim
        # out = nn.Flatten()(out)

        out = self.classifier(out)
        return out

