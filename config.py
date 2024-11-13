import argparse


def parse_option():
    parser = argparse.ArgumentParser('Supervised Contrastive Learning with Config and CLI')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--temp', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency for checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--dataset', type=str, default='./data', help='Dataset to use')
    parser.add_argument('--model_type', type=str, default='resnet34',
                        help='Model type (e.g., resnet34, ResNeXt101, WideResNet)')
    parser.add_argument('--augmentation', type=str, default='basic',
                        help='Data augmentation method (e.g., basic, advanced)')

    parser.add_argument('--class_name', type=str, default='default_class', help='Class name for dataset')
    parser.add_argument('--action_type', type=str, default='norm-train', help='Action type (e.g., norm-train)')

    args = parser.parse_args()

    # 直接将 argparse 的解析结果转换为字典形式
    opt = vars(args)

    return opt


def get_config():
    return parse_option()
