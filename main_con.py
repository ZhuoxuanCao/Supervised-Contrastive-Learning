import os
import torch
from Contrastive_Learning import config_con
from Contrastive_Learning import train, set_loader, set_model, create_scheduler, LARS, save_model


def ensure_dir_exists(path):
    """Ensure that the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def main():
    # 从配置文件获取配置
    opt = config_con.get_config()

    # 确保保存目录存在
    ensure_dir_exists(opt['model_save_dir'])

    # 设置设备
    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() else "cpu")
    opt['device'] = device

    # 创建数据加载器和模型
    train_loader = set_loader(opt)
    model, criterion, device = set_model(opt)

    # 优化器和调度器
    optimizer = LARS(
        model.parameters(),
        lr=opt['learning_rate'],
        momentum=0.9,
        weight_decay=1e-4,
        eta=0.001,
        epsilon=1e-8
    )
    scheduler = create_scheduler(optimizer, warmup_epochs=5, total_epochs=opt['epochs'])

    # 训练循环
    best_loss = float('inf')
    for epoch in range(opt['epochs']):
        print(f"Epoch [{epoch + 1}/{opt['epochs']}]")
        epoch_loss = train(train_loader, model, criterion, optimizer, opt, device, epoch)

        # 更新调度器
        scheduler.step()

        print(f"Train Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # 保存当前最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_root = "./saved_models/pretraining"  # 预训练模型的根目录
            save_model(model, opt, epoch, epoch_loss, save_root)

    print(f"Training complete. Best loss: {best_loss:.4f}")



if __name__ == "__main__":
    main()

