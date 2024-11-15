import torch
import torch.nn as nn

class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=8, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        D = int(out_channels * (base_width / 64))  # 每组的宽度
        self.conv1 = nn.Conv2d(in_channels, D * cardinality, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * cardinality)
        self.conv2 = nn.Conv2d(D * cardinality, D * cardinality, kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)  # 分组卷积
        self.bn2 = nn.BatchNorm2d(D * cardinality)
        self.conv3 = nn.Conv2d(D * cardinality, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality=32, base_width=8, num_classes=10):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, cardinality=cardinality, base_width=base_width)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, cardinality=cardinality, base_width=base_width)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, cardinality=cardinality, base_width=base_width)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, cardinality=cardinality, base_width=base_width)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride, cardinality, base_width):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, cardinality, base_width, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, cardinality=cardinality, base_width=base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 定义函数创建 ResNeXt101_32x8d
def ResNeXt101_32x8d(num_classes=10):
    return ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], cardinality=32, base_width=8, num_classes=num_classes)

# 测试模型结构（如果您在本地运行）
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNeXt101_32x8d().to(device)
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print("Model architecture:\n", model)
