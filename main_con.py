import os
import torch
from Contrastive_Learning import config_con  # 配置模块
from Contrastive_Learning import train, set_loader, set_model, create_scheduler, LARS


def ensure_dir_exists(path):
    """Ensure that the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def save_pretrained_model(model, save_dir, epoch, batch_size, opt, last_save_path=None):
    """
    Save the pre-trained model with additional metadata in the filename.
    """
    # 动态构建文件名，加入数据集和模型相关信息
    dataset_info = opt['dataset_name']
    model_info = f"{opt['model_type']}_{dataset_info}_feat{opt.get('feature_dim', 128)}_{opt['loss_type']}"
    save_path = os.path.join(save_dir, f"{model_info}_epoch{epoch}_batch{batch_size}.pth")

    # 保存模型状态和配置信息
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": opt,  # 保存训练时的超参数配置
    }, save_path)
    print(f"New best model saved to {save_path}")

    # 删除上一次保存的模型文件（如果存在）
    if last_save_path and os.path.exists(last_save_path):
        os.remove(last_save_path)
        print(f"Deleted previous model: {last_save_path}")

    return save_path


def main():
    opt = config_con.get_config()

    pretrain_dir = os.path.join("saved_models", "pretraining", opt["model_type"])
    ensure_dir_exists(pretrain_dir)

    train_loader, _ = set_loader(opt)
    model, criterion, device = set_model(opt)

    optimizer = LARS(
        model.parameters(),
        lr=opt['learning_rate'],
        momentum=0.9,
        weight_decay=1e-4,
        eta=0.001,
        epsilon=1e-8
    )

    scheduler = create_scheduler(optimizer, warmup_epochs=5, total_epochs=opt['epochs'])

    best_loss = float("inf")
    best_epoch = -1
    last_save_path = None

    for epoch in range(opt['epochs']):
        print(f"Epoch [{epoch + 1}/{opt['epochs']}]")
        epoch_loss = train(train_loader, model, criterion, optimizer, opt, device)
        scheduler.step()

        print(f"Train Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            last_save_path = save_pretrained_model(model, pretrain_dir, best_epoch, opt['batch_size'], opt, last_save_path)

    print(f"Training complete. Best model at epoch {best_epoch} with loss {best_loss:.4f}.")


if __name__ == "__main__":
    main()
