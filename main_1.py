import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from tensorboard_logger import Logger
from torch.utils.tensorboard import SummaryWriter


from losses.SupOut import SupConLoss_out
from models.model_1 import ResModel
from models.model_2_resnet34 import ResNet34
from models.model_3_ResNeXt101_32x8d import ResNeXt101_32x8d

from data_augmentation.data_augmentation_1 import TwoCropTransform, get_base_transform

# 定义参数解析
def parse_option():
    parser = argparse.ArgumentParser('Supervised Contrastive Learning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--temp', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency for checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints', help='Directory to save models')
    return parser.parse_args()


# 加载数据集和数据增强
def set_loader(opt):
    transform = TwoCropTransform(get_base_transform())  # 应用 TwoCropTransform
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    return train_loader


# 设置模型和损失函数
def set_model(opt):
    model = ResNet34().cuda()
    criterion = SupConLoss_out(temperature=opt.temp).cuda()
    return model, criterion


# 学习率调整
def adjust_learning_rate(optimizer, epoch, opt):
    if epoch in [4, 8, 12]:  # 可以按需修改衰减周期
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5


# 训练函数
def train(train_loader, model, criterion, optimizer, opt, writer):
    model.train()
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(optimizer, epoch, opt)
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            # 确保 inputs 是包含两个视图的列表
            if isinstance(inputs, list) and len(inputs) == 2:
                inputs = torch.cat([inputs[0], inputs[1]], dim=0).cuda()  # 拼接两个增强视图，维度为 [2*batch_size, 3, 32, 32]
            else:
                raise ValueError("Expected 'inputs' to be a list of two views, but got something else.")

            labels = labels.cuda()
            optimizer.zero_grad()

            # 获取特征并重新组织为 [batch_size, n_views, feature_dim]
            features = model(inputs)
            f1, f2 = torch.split(features, features.size(0) // 2, dim=0)
            features = torch.stack([f1, f2], dim=1)

            # 计算损失
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch}/{opt.epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        writer.add_scalar('Epoch Loss', running_loss / len(train_loader), epoch)
    return running_loss / len(train_loader)


def main():
    opt = parse_option()
    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)
    if not os.path.isdir(opt.model_save_dir):
        os.makedirs(opt.model_save_dir)

    # 设置日志记录
    writer = SummaryWriter(log_dir=opt.log_dir)

    # 加载数据和模型
    train_loader = set_loader(opt)
    model, criterion = set_model(opt)

    # 设置优化器
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-4)

    # 训练模型
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(optimizer, epoch, opt)
        epoch_loss = train(train_loader, model, criterion, optimizer, opt, writer)

        # 记录损失和学习率到 TensorBoard
        writer.add_scalar('loss', epoch_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # 保存模型
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.model_save_dir, f'ckpt_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_file)
            print(f'Model saved to {save_file}')

    # 保存最后的模型
    save_file = os.path.join(opt.model_save_dir, 'last.pth')
    torch.save(model.state_dict(), save_file)
    writer.close()


if __name__ == '__main__':
    main()
