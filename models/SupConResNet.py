import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConResNet(nn.Module):
    """
    A ResNet model extended with a projection head for contrastive learning.
    """
    def __init__(self, base_model, feature_dim=128, dim_in=None):
        """
        Initialize the SupConResNet model.
        Args:
            base_model (nn.Module): Base ResNet model (e.g., ResNet34, ResNet50).
            feature_dim (int): Output feature dimension of the projection head.
            dim_in (int): Input feature dimension for the projection head.
                          If None, it must be provided when using ResNet variants.
        """
        super(SupConResNet, self).__init__()
        self.encoder = base_model  # Base ResNet backbone

        if dim_in is None:
            raise ValueError("The input feature dimension (dim_in) must be specified for the projection head.")

        print(f"Encoder output feature dimension: {dim_in}")  # Debugging line

        # Projection head (MLP)
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feature_dim)
        )

    def forward(self, x):
        """
        Forward pass for the SupConResNet.
        Args:
            x: Input data, can be a list for multi-view contrastive learning or a tensor for single view.
        Returns:
            Normalized features for contrastive loss.
        """
        if isinstance(x, list) and len(x) == 2:  # Multi-view input
            feat1 = self.encoder(x[0])
            feat2 = self.encoder(x[1])
            feat1 = F.normalize(self.head(feat1), dim=1)  # Normalize
            feat2 = F.normalize(self.head(feat2), dim=1)  # Normalize
            return [feat1, feat2]

        # Single view input
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)  # Normalize
        return feat

def determine_feature_dim(model):
    """
    Dynamically determine the output feature dimension of a model.
    """
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 32, 32)  # Dummy input for CIFAR-10
            features = model(dummy_input)
            return features.shape[1]
    finally:
        model.train()  # Always restore training mode, even if an exception occurs

def SupConResNetFactory(base_model_func, feature_dim=128):
    """
    Factory function to create SupConResNet models based on ResNet variants.
    """
    base_model = base_model_func()
    base_model.fc = nn.Identity()  # 移除分类头
    dim_in = determine_feature_dim(base_model)  # 动态确定特征维度
    return SupConResNet(base_model, feature_dim, dim_in)


# def SupConResNetFactory(base_model_func, feature_dim=128, dim_in=None):
#     """
#     Factory function to create SupConResNet models based on ResNet variants.
#     Args:
#         base_model_func (callable): A callable that returns a ResNet model (e.g., ResNet50, ResNet101).
#         feature_dim (int): Output feature dimension of the projection head.
#         dim_in (int): Input feature dimension for the projection head.
#                       Defaults to 512 for ResNet34 and 2048 for ResNet50/101/200.
#     Returns:
#         SupConResNet: A SupConResNet model with a projection head.
#     """
#     # Create the base model, ensure base_model_func is instantiated correctly
#     base_model = base_model_func() if callable(base_model_func) else base_model_func
#
#     base_model.fc = nn.Identity()  # Remove classification head
#
#     if dim_in is None:
#         # Infer feature dimension based on ResNet variant
#         # dim_in = 512 if base_model_func.__name__ == "ResNet34" else 2048
#         dim_in = 512 if base_model_func.__name__ == "ResNet34" else 2048
#     base_model.fc = nn.Identity()  # Remove the classification head
#     return SupConResNet(base_model, feature_dim, dim_in)
