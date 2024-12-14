import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models import ResModel, ResNet34, ResNeXt101_32x8d, WideResNet_28_10, ResNet50, ResNet101, ResNet200, \
    CSPDarknet53Classifier, SupConResNetFactory
import os


def ensure_dir_exists(path):
    """
    Ensure that the directory exists. If not, create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def save_best_model(classifier, save_path, last_save_path):
    """
    Save the best classifier model and delete the previous one.
    """
    # 如果当前保存路径和上一次保存路径相同，则跳过删除
    if save_path == last_save_path:
        print(f"Saving new model to {save_path}, skipping deletion of identical path.")
    else:
        # 删除上一次保存的模型文件（如果存在）
        if last_save_path and os.path.exists(last_save_path):
            os.remove(last_save_path)
            print(f"Deleted previous model: {last_save_path}")

    # 保存新模型
    torch.save(classifier.state_dict(), save_path)
    print(f"New best model saved to {save_path}")

    return save_path  # 返回当前保存路径



def train_classifier(train_loader, val_loader, model, classifier, optimizer, criterion, device, epochs=10, save_dir="./saved_models", model_type="ResNet50"):
    model.eval()  # Freeze feature extractor
    classifier.train()  # Train classification head

    best_accuracy = 0.0  # 记录验证集上最高的准确率
    last_save_path = None  # 记录上一次保存的模型路径
    ensure_dir_exists(save_dir)

    for epoch in range(epochs):
        # Training loop
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

            # Calculate training accuracy
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy * 100:.2f}%")

        # Validation loop
        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_features = model.encoder(val_inputs)
                val_outputs = classifier(val_features)
                _, val_predicted = val_outputs.max(1)
                val_correct += (val_predicted == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_accuracy = val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Save the best model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_path = os.path.join(save_dir, f"{model_type}_best_classifier.pth")
            last_save_path = save_best_model(classifier, save_path, last_save_path)

    print(f"Training complete. Best model saved with validation accuracy: {best_accuracy * 100:.2f}%")


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
    parser.add_argument("--save_dir", type=str, default="./saved_models/classification", help="Directory to save the best classifier")
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

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

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

    # 加载预训练模型
    checkpoint = torch.load(args.pretrained_model)  # 加载包含多个键的字典
    model.load_state_dict(checkpoint["model_state_dict"])  # 只加载模型权重
    model = model.to(device)

    # Define classification head
    classifier = nn.Linear(2048 if args.model_type != "ResNet34" else 512, 10)  # CIFAR-10 has 10 classes
    classifier = classifier.to(device)

    # Define optimizer and loss
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train classifier
    train_classifier(train_loader, val_loader, model, classifier, optimizer, criterion, device, epochs=args.epochs, save_dir=args.save_dir, model_type=args.model_type)


if __name__ == "__main__":
    main()

# python train_classifier.py --model_type ResNet34 --pretrained_model ./saved_models/pretraining/ResNet34/ResNet34_epoch_3.pth --save_dir ./saved_models/classification --batch_size 32 --epochs 5 --learning_rate 0.001 --dataset_name cifar10 --dataset ./data --gpu 0
