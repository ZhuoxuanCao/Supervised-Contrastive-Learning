import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.SupConResNet import SupConResNetFactory
from models import ResModel, ResNet34, ResNeXt101_32x8d, WideResNet_28_10, ResNet50, ResNet101, ResNet200, \
    CSPDarknet53Classifier, SupConResNetFactory
from utils import save_model


def train_classifier(train_loader, model, classifier, optimizer, criterion, device, epochs=10):
    model.eval()  # Freeze feature extractor
    classifier.train()  # Train classification head

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Extract features (freeze encoder)
            with torch.no_grad():
                features = model.encoder(inputs)

            # Forward pass through classifier
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%")

    return classifier


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a classification head on top of a frozen feature extractor")
    parser.add_argument("--model_type", type=str, default="ResNet50", help="Model type (ResNet50, ResNet34, ResNet101)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dataset_name", type=str, default="cifar10", help="Dataset name (cifar10, cifar100)")
    parser.add_argument("--dataset", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to pre-trained SupConResNet")
    parser.add_argument("--save_classifier", type=str, default="classifier.pth", help="Path to save the classifier")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (default: 0)")

    args = parser.parse_args()

    # Device configuration
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Dataset loading
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform)
    elif args.dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=args.dataset, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

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

    # Define classification head
    classifier = nn.Linear(2048 if args.model_type != "ResNet34" else 512, 10)  # CIFAR-10 has 10 classes
    classifier = classifier.to(device)

    # Define optimizer and loss
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train classifier
    classifier = train_classifier(train_loader, model, classifier, optimizer, criterion, device, epochs=args.epochs)

    # Save classifier
    save_model(classifier, args.save_classifier)
    print(f"Classifier saved to {args.save_classifier}")


if __name__ == "__main__":
    main()
