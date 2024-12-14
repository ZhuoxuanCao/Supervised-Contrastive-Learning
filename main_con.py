# import os
# import torch
# from Contrastive_Learning import config_con  # 配置模块
# from Contrastive_Learning import train, set_loader, set_model
#
# def main():
#     # 从配置文件获取配置
#     opt = config_con.get_config()
#
#     # 创建日志目录和模型保存目录
#     log_dir = opt['log_dir']
#     model_save_dir = opt['model_save_dir']
#
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#
#     if not os.path.exists(model_save_dir):
#         os.makedirs(model_save_dir)
#
#     # 创建数据加载器和模型
#     train_loader, test_loader = set_loader(opt)
#     model, criterion, device = set_model(opt)
#
#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=opt['learning_rate'],
#         momentum=0.9,
#         weight_decay=1e-4,
#     )
#
#     # 训练循环
#     for epoch in range(opt['epochs']):
#         print(f"Epoch [{epoch + 1}/{opt['epochs']}]")
#         epoch_loss = train(train_loader, model, criterion, optimizer, opt, device)
#         print(f"Train Loss: {epoch_loss:.4f}")
#
#         # 保存检查点
#         if (epoch + 1) % opt['save_freq'] == 0 or (epoch + 1) == opt['epochs']:
#             checkpoint_path = os.path.join(opt['model_save_dir'], f"checkpoint_epoch_{epoch + 1}.pth")
#             torch.save(model.state_dict(), checkpoint_path)
#             print(f"Checkpoint saved to {checkpoint_path}")
#
#     print("Training complete.")
#
# if __name__ == "__main__":
#     main()


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


def save_pretrained_model(model, save_dir, epoch, opt):
    """
    Save the pre-trained model and additional metadata.
    """
    save_path = os.path.join(save_dir, f"{opt['model_type']}_epoch_{epoch}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": opt,  # 保存训练时的超参数配置
    }, save_path)
    print(f"New best model saved to {save_path}")


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

    # 训练循环
    for epoch in range(opt['epochs']):
        print(f"Epoch [{epoch + 1}/{opt['epochs']}]")
        epoch_loss = train(train_loader, model, criterion, optimizer, opt, device)
        print(f"Train Loss: {epoch_loss:.4f}")

        # 检查当前损失是否为最低
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            save_pretrained_model(model, pretrain_dir, best_epoch, opt)

    print(f"Training complete. Best model at epoch {best_epoch} with loss {best_loss:.4f}.")

if __name__ == "__main__":
    main()
