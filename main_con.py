import os
import torch
import logging  # 引入日志模块
from Contrastive_Learning import config_con
from Contrastive_Learning import train, set_loader, set_model, create_scheduler, LARS, save_best_model

def ensure_dir_exists(path):
    """Ensure that the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def setup_logging(log_dir):
    """
    配置日志记录功能，如果日志路径不存在，则自动创建。
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 自动创建目录
        print(f"Created log directory: {log_dir}")

    log_file = os.path.join(log_dir, "training.log")  # 定义日志文件路径

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(levelname)s - %(message)s",  # 定义日志格式
        handlers=[
            logging.FileHandler(log_file),  # 将日志写入文件
            logging.StreamHandler()        # 同时在控制台打印
        ]
    )
    logging.info("Logging initialized. Logs will be saved to: %s", log_file)


def main():
    # 从配置文件获取配置
    opt = config_con.get_config()

    # 设置日志功能
    setup_logging(opt['log_dir'])

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
        epsilon=1e-8,
        min_lr=1e-6  # 添加最小学习率下限
    )
    scheduler = create_scheduler(optimizer, warmup_epochs=5, total_epochs=opt['epochs'])

    # 训练循环
    best_loss = float('inf')  # 初始化最佳损失
    last_save_path = None  # 初始化最后保存的路径
    save_root = "./saved_models/pretraining"  # 预训练模型的根目录

    for epoch in range(opt['epochs']):
        logging.info(f"Epoch [{epoch + 1}/{opt['epochs']}] started.")
        print(f"Epoch [{epoch + 1}/{opt['epochs']}]")
        epoch_loss = train(train_loader, model, criterion, optimizer, opt, device, epoch)

        # 更新调度器
        scheduler.step()

        logging.info(f"Epoch [{epoch + 1}/{opt['epochs']}]: Train Loss: {epoch_loss:.4f}, "
                     f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        print(f"Train Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # 保存当前性能最佳的模型
        best_loss, last_save_path = save_best_model(model, opt, epoch, epoch_loss, save_root, best_loss, last_save_path)

    logging.info(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Training complete. Best loss: {best_loss:.4f}")




if __name__ == "__main__":
    main()

