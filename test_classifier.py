import os
import torch
import torch.nn as nn
import argparse
from sympy import false
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from losses import SupConLoss_in, SupConLoss_out, CrossEntropyLoss

from models import ResModel, ResNet34, ResNeXt101_32x8d, WideResNet_28_10, ResNet50, ResNet101, ResNet200, \
    CSPDarknet53, SupConResNetFactory

from data_augmentation.data_augmentation_1 import TwoCropTransform, get_base_transform

from utils import load_model


from torchvision import datasets, transforms

def test_classifier(test_loader, model, classifier, device):
    """
    Evaluate the classification head on a frozen feature extractor.
    """
    model.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Extract features
            features = model.encoder(inputs)

            # Classify
            outputs = classifier(features)
            _, predicted = outputs.max(1)

            # Calculate accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Test a classification head on top of a frozen feature extractor")
    parser.add_argument("--model_type", type=str, default="ResNet50", help="Model type (ResNet50, ResNet34, ResNet101)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dataset_name", type=str, default="cifar10", help="Dataset name (cifar10, cifar100)")
    parser.add_argument("--dataset", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to pre-trained SupConResNet")
    parser.add_argument("--classifier", type=str, required=True, help="Path to trained classifier")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (default: 0)")

    args = parser.parse_args()

    # Device configuration
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Dataset loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform)
    elif args.dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    model_dict = {
        "ResNet50": lambda: ResNet50(),
        "ResNet34": lambda: ResNet34(),
        "ResNet101": lambda: ResNet101(),
    }
    base_model_func = model_dict.get(args.model_type)
    if base_model_func is None:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model = SupConResNetFactory(base_model_func, feature_dim=128)
    model.load_state_dict(torch.load(args.pretrained_model))
    model = model.to(device)

    # Load classifier
    classifier = load_model(args.classifier)
    classifier = classifier.to(device)

    # Test classifier
    accuracy = test_classifier(test_loader, model, classifier, device)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
