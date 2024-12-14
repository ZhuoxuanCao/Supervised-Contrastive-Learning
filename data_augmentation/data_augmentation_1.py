from torchvision import transforms

class TwoCropTransform:
    """Create two crops of the same image for contrastive learning."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

def get_base_transform(input_resolution=32):
    """
    定义基本的数据增强组合，支持动态输入分辨率。
    Args:
        input_resolution (int): 输入图像的分辨率（默认32）。
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_resolution, scale=(0.2, 1.0)),  # 动态分辨率
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # 颜色抖动
        transforms.RandomGrayscale(p=0.2),           # 随机灰度
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

