import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from losses import SupConLoss_in, SupConLoss_out, CrossEntropyLoss
from models import ResNet34, ResNet50, ResNet101, ResNet200, CSPDarknet53, SupConResNetFactory, SupConResNetFactory_CSPDarknet53
from data_augmentation import TwoCropTransform, get_base_transform
from torchvision import datasets
from torch.optim.optimizer import Optimizer




# LARS 优化器实现
class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training."""

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, eta=0.001, epsilon=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, epsilon=epsilon)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                param_norm = torch.norm(p)
                grad_norm = torch.norm(grad)

                if param_norm > 0 and grad_norm > 0:
                    adaptive_lr = group['eta'] * param_norm / (grad_norm + group['epsilon'])
                else:
                    adaptive_lr = 1.0

                p.add_(grad, alpha=-group['lr'] * adaptive_lr)

        return loss


def set_loader(opt):
    if opt['augmentation'] == 'basic':
        transform = TwoCropTransform(get_base_transform())
    else:
        raise ValueError(f"Unknown augmentation type: {opt['augmentation']}")

    # 根据数据集名称选择数据集
    if opt['dataset_name'] == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt['dataset'], train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root=opt['dataset'], train=False, download=True, transform=transform)
    elif opt['dataset_name'] == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt['dataset'], train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR100(root=opt['dataset'], train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {opt['dataset_name']}")

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=2)
    return train_loader, test_loader


def set_model(opt):
    model_dict = {
        'ResNet34': lambda: ResNet34(num_classes=opt['num_classes']),
        'ResNet50': lambda: ResNet50(num_classes=opt['num_classes']),
        'ResNet101': lambda: ResNet101(num_classes=opt['num_classes']),
        'ResNet200': lambda: ResNet200(num_classes=opt['num_classes']),
        'CSPDarknet53': lambda: CSPDarknet53(num_classes=10, input_resolution=32, num_blocks=[1, 2, 8, 8, 4]),
    }
    base_model_func = model_dict.get(opt['model_type'])

    if base_model_func is None:
        raise ValueError(f"Unknown model type: {opt['model_type']}")

    # 根据模型类型选择使用哪个 SupConResNetFactory
    if opt['model_type'] == 'CSPDarknet53':
        # 使用新的 SupConResNetFactory_CSPDarknet53
        model = SupConResNetFactory_CSPDarknet53(
            base_model_func=base_model_func,
            feature_dim=opt.get("feature_dim", 128),
        )
    else:
        # 使用旧的 SupConResNetFactory
        model = SupConResNetFactory(
            base_model_func=base_model_func,
            feature_dim=opt.get("feature_dim", 128),
        )

    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() and opt['gpu'] is not None else "cpu")
    model = model.to(device)

    # 根据 loss_type 参数选择损失函数
    if opt['loss_type'] == 'supout':
        criterion = SupConLoss_out(temperature=opt['temp']).to(device)
    elif opt['loss_type'] == 'cross_entropy':
        criterion = CrossEntropyLoss().to(device)
    elif opt['loss_type'] == 'supin':
        criterion = SupConLoss_in().to(device)
    else:
        raise ValueError(f"Unknown loss type: {opt['loss_type']}")

    return model, criterion, device


def create_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    创建 Warmup + 余弦退火学习率调度器
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * math.pi))
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def train(train_loader, model, criterion, optimizer, opt, device):
    model.train()
    running_loss = 0.0

    for step, (inputs, labels) in enumerate(train_loader):
        # 数据预处理
        if isinstance(inputs, list) and len(inputs) == 2:
            inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(device)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        features = model(inputs)

        # 对比损失特征处理
        f1, f2 = torch.split(features, features.size(0) // 2, dim=0)
        contrastive_features = torch.stack([f1, f2], dim=1)  # [batch_size // 2, 2, feature_dim]

        # 检查特征和标签是否匹配
        if contrastive_features.size(0) != labels.size(0):
            labels = labels[:contrastive_features.size(0)]  # 截取标签以匹配特征

        # 计算损失
        loss = criterion(contrastive_features, labels)
        loss.backward()
        optimizer.step()

        # 累加损失
        running_loss += loss.item()

        # 打印训练进度
        if (step + 1) % 100 == 0:
            print(f"Step [{step + 1}/{len(train_loader)}], Loss: {running_loss / (step + 1):.4f}")

    # 返回损失
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


