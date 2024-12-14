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


# 数据加载器
def set_loader(opt):
    transform = TwoCropTransform(get_base_transform(opt['input_resolution']))
    if opt['dataset_name'] == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt['dataset'], train=True, download=False, transform=transform)
    elif opt['dataset_name'] == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt['dataset'], train=True, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {opt['dataset_name']}")

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=2)
    return train_loader


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
            feature_dim=opt['feature_dim'],
        )
    else:
        # 使用旧的 SupConResNetFactory
        model = SupConResNetFactory(
            base_model_func=base_model_func,
            feature_dim=opt['feature_dim'],
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
        return 0.5 * (1 + math.cos((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * math.pi))
    return LambdaLR(optimizer, lr_lambda)


# 模型保存
def save_model(model, opt, epoch, loss, save_root):
    """
    保存模型到指定的文件夹，按模型类型分类管理。
    Args:
        model: 需要保存的模型。
        opt: 配置字典，包含超参数信息。
        epoch: 当前训练轮次。
        loss: 当前训练轮次的损失。
        save_root: 保存模型的根目录。
    """
    # 构造模型类型的子目录
    model_dir = os.path.join(save_root, opt['model_type'])

    # 确保目标目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # 自动创建目录
        print(f"Created directory: {model_dir}")

    # 构造保存路径
    save_path = os.path.join(
        model_dir,
        f"{opt['model_type']}_{opt['dataset_name']}_feat{opt['feature_dim']}_res{opt['input_resolution']}_epoch{epoch}_loss{loss:.4f}.pth"
    )

    # 保存模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": opt  # 保存超参数配置
    }, save_path)

    print(f"Model saved to {save_path}")


# 训练函数
def train(train_loader, model, criterion, optimizer, opt, device, epoch=None):
    model.train()
    running_loss = 0.0
    total_steps = len(train_loader)  # 总步数

    # 打印阶段信息
    print(f"Start training: Epoch [{epoch + 1}/{opt['epochs']}]") if epoch is not None else None

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

        # 打印阶段性训练进度信息
        if (step + 1) % 100 == 0 or (step + 1) == total_steps:
            avg_loss = running_loss / (step + 1)
            print(f"Step [{step + 1}/{total_steps}], Loss: {avg_loss:.4f}")

    # 返回损失
    epoch_loss = running_loss / len(train_loader)

    # 阶段性输出
    if epoch is not None and (epoch + 1) % 5 == 0:
        print(f"--- Summary for Epoch [{epoch + 1}] ---")
        print(f"    Average Loss: {epoch_loss:.4f}")
        print(f"    Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    return epoch_loss



