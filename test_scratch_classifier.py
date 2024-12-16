import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import ResNet34  # 确保路径正确
import os

# 加载 CIFAR-10 测试集
def load_test_data(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return testloader

# 测试模型性能
def test_model(model_path, device):
    # 加载模型
    model = ResNet34(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式

    # 加载测试数据
    testloader = load_test_data()

    # 测试逻辑
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(testloader)
    print(f"Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./saved_models/classification/scratch/ResNet34_cifar10_batch16_valAcc80.52_20241216-234330.pth"

    # 测试模型
    test_model(model_path, device)
