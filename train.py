import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from losses.SupOut import SupConLoss_out

from models.model_1 import ResModel
from models.model_2_resnet34 import ResNet34
from models.model_3_ResNeXt101_32x8d import ResNeXt101_32x8d
from models.model_4_WideResNet_28_10 import WideResNet_28_10

from data_augmentation.data_augmentation_1 import TwoCropTransform, get_base_transform
from torchvision import datasets





def set_loader(opt):
    if opt['augmentation'] == 'basic':
        transform = TwoCropTransform(get_base_transform())
    # elif opt['augmentation'] == 'advanced':
    #     transform = TwoCropTransform(get_advanced_transform())
    else:
        raise ValueError(f"Unknown augmentation type: {opt['augmentation']}")

    train_dataset = datasets.CIFAR10(root=opt['dataset'], train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=2)
    return train_loader

def set_model(opt):
    if opt['model_type'] == 'resnet34':
        model = ResNet34().cuda() if opt['gpu'] is not None else ResNet34()
    elif opt['model_type'] == 'ResNeXt101':
        model = ResNeXt101_32x8d().cuda() if opt['gpu'] is not None else ResNeXt101_32x8d()
    elif opt['model_type'] == 'WideResNet':
        model = WideResNet_28_10().cuda() if opt['gpu'] is not None else WideResNet_28_10()
    elif opt['model_type'] == 'resnet_HikVision':  # 自定义模型
        model = ResModel().cuda() if opt['gpu'] is not None else ResModel()
    else:
        raise ValueError(f"Unknown model type: {opt['model_type']}")

    criterion = SupConLoss_out(temperature=opt['temp']).cuda() if opt['gpu'] is not None else SupConLoss_out(temperature=opt['temp'])
    return model, criterion

def adjust_learning_rate(optimizer, epoch, opt):
    if epoch in [4, 8, 12]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5

def train(train_loader, model, criterion, optimizer, opt, writer):
    model.train()
    for epoch in range(1, opt['epochs'] + 1):
        adjust_learning_rate(optimizer, epoch, opt)
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            if isinstance(inputs, list) and len(inputs) == 2:
                inputs = torch.cat([inputs[0], inputs[1]], dim=0).cuda() if opt['gpu'] is not None else torch.cat([inputs[0], inputs[1]], dim=0)
            labels = labels.cuda() if opt['gpu'] is not None else labels
            optimizer.zero_grad()
            features = model(inputs)
            f1, f2 = torch.split(features, features.size(0) // 2, dim=0)
            features = torch.stack([f1, f2], dim=1)
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f'Epoch [{epoch}/{opt["epochs"]}], Step [{step + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        writer.add_scalar('Epoch Loss', running_loss / len(train_loader), epoch)
    return running_loss / len(train_loader)
