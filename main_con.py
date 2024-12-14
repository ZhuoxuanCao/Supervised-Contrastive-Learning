import os
import torch
from Contrastive_Learning import config_con  # 配置模块
from Contrastive_Learning import train, set_loader, set_model


def ensure_dir_exists(path):
    """
    Ensure that the directory exists. If not, create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def save_pretrained_model(model, save_dir, epoch, opt, last_save_path=None):
    """
    Save the pre-trained model and additional metadata. Optionally remove the previous model.
    """
    save_path = os.path.join(save_dir, f"{opt['model_type']}_epoch_{epoch}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": opt,  # 保存训练时的超参数配置
    }, save_path)
    print(f"New best model saved to {save_path}")

    # 删除上一次保存的模型文件（如果存在）
    if last_save_path and os.path.exists(last_save_path):
        os.remove(last_save_path)
        print(f"Deleted previous model: {last_save_path}")

    return save_path  # 返回当前保存的文件路径


def main():
    # 从配置文件获取配置
    opt = config_con.get_config()

    # 创建专门的预训练模型保存目录
    pretrain_dir = os.path.join("saved_models", "pretraining", opt["model_type"])
    ensure_dir_exists(pretrain_dir)

    # 创建数据加载器和模型
    train_loader, test_loader = set_loader(opt)
    model, criterion, device = set_model(opt)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt['learning_rate'],
        momentum=0.9,
        weight_decay=1e-4,
    )

    # 初始化最小损失和最佳模型信息
    best_loss = float("inf")  # 初始值为正无穷
    best_epoch = -1
    last_save_path = None  # 用于记录上一次保存的模型路径

    # 训练循环
    for epoch in range(opt['epochs']):
        print(f"Epoch [{epoch + 1}/{opt['epochs']}]")
        epoch_loss = train(train_loader, model, criterion, optimizer, opt, device)
        print(f"Train Loss: {epoch_loss:.4f}")

        # 检查当前损失是否为最低
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            last_save_path = save_pretrained_model(model, pretrain_dir, best_epoch, opt, last_save_path)

    print(f"Training complete. Best model at epoch {best_epoch} with loss {best_loss:.4f}.")

if __name__ == "__main__":
    main()
