# 问题：学习率调整逻辑，是否还能优化

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models import ResNet34, ResNet50, ResNet101, ResNet200, SupConResNetFactory
import os
from tqdm import tqdm  # 用于显示进度条
import datetime

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def save_best_model(backbone, classifier, save_path, last_save_path):
    if save_path == last_save_path:
        print(f"Saving new model to {save_path}, skipping deletion of identical path.")
    else:
        if last_save_path and os.path.exists(last_save_path):
            os.remove(last_save_path)
            print(f"Deleted previous model: {last_save_path}")

    # 保存 Backbone 和分类头的权重
    torch.save({
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
    }, save_path)
    print(f"New best model saved to {save_path}")
    return save_path


def train_classifier(train_loader, val_loader, model, classifier, optimizer, scheduler, criterion, device, epochs=10,
                     save_dir="./saved_models", model_type="ResNet50", batch_size=64, use_pretrained=True, dataset_name="cifar10"):
    if use_pretrained:
        model.eval()
    else:
        model.train()
    classifier.train()

    best_accuracy = 0.0
    last_save_path = None
    ensure_dir_exists(save_dir)

    try:
        for epoch in range(epochs):
            print(f"Epoch [{epoch + 1}/{epochs}]")

            running_loss = 0.0
            correct = 0
            total = 0
            batch_losses = []
            batch_accuracies = []

            model.train()
            train_bar = tqdm(train_loader, desc="Training", leave=False)
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                if use_pretrained:
                    with torch.no_grad():
                        features = model.encoder(inputs)
                else:
                    features = model.encoder(inputs)

                outputs = classifier(features)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                running_loss += loss.item()
                batch_losses.append(loss.item())
                batch_accuracies.append((predicted == labels).float().mean().item())

                train_bar.set_postfix(loss=loss.item(), acc=batch_accuracies[-1] * 100)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy * 100:.2f}%")
            print(f"  Batch Loss: min={min(batch_losses):.4f}, max={max(batch_losses):.4f}, mean={epoch_loss:.4f}")
            print(
                f"  Batch Accuracy: min={min(batch_accuracies) * 100:.2f}%, max={max(batch_accuracies) * 100:.2f}%, mean={epoch_accuracy * 100:.2f}%")

            classifier.eval()
            val_correct = 0
            val_total = 0
            val_running_loss = 0.0
            val_bar = tqdm(val_loader, desc="Validating", leave=False)
            with torch.no_grad():
                for val_inputs, val_labels in val_bar:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_features = model.encoder(val_inputs)
                    val_outputs = classifier(val_features)
                    val_loss = criterion(val_outputs, val_labels)

                    _, val_predicted = val_outputs.max(1)
                    val_correct += (val_predicted == val_labels).sum().item()
                    val_total += val_labels.size(0)
                    val_running_loss += val_loss.item()

            val_loss = val_running_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(save_dir,
                                         f"{model_type}_{dataset_name}_batch{batch_size}_valAcc{val_accuracy * 100:.2f}_{timestamp}.pth")
                last_save_path = save_best_model(model, classifier, save_path, last_save_path)

            scheduler.step()

        print(f"Training complete. Best model saved with validation accuracy: {best_accuracy * 100:.2f}%")
    except Exception as e:
        print(f"Error during training: {e}")
        raise



def main():
    parser = argparse.ArgumentParser(description="Train a classification head on top of a frozen feature extractor")
    parser.add_argument("--model_type", type=str, default="ResNet50", help="Model type (ResNet50, ResNet34, ResNet101, ResNet200)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dataset_name", type=str, default="cifar10", help="Dataset name (cifar10, cifar100, imagenet)")
    parser.add_argument("--dataset", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to pre-trained SupConResNet (required if --use_pretrained)")
    parser.add_argument("--save_dir", type=str, default="./saved_models/classification/pretrained", help="Directory to save the best classifier")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (default: 0)")
    # parser.add_argument("--use_pretrained", action="store_true", help="Use a pre-trained SupConResNet (default: False)")
    # parser.add_argument("--no_pretrained", dest="use_pretrained", action="store_false", help="Do not use pre-trained weights")
    parser.set_defaults(use_pretrained=True)

    args = parser.parse_args()

    if args.use_pretrained and not args.pretrained_model:
        parser.error("--pretrained_model is required when --use_pretrained is True.")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.dataset_name == "cifar10":
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform)
        num_classes = 10
    elif args.dataset_name == "cifar100":
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR100(root=args.dataset, train=True, download=True, transform=transform)
        num_classes = 100
    elif args.dataset_name == "imagenet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(root=os.path.join(args.dataset, "train"), transform=transform)
        num_classes = 1000
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model_dict = {
        "ResNet34": lambda: ResNet34(),
        "ResNet50": lambda: ResNet50(),
        "ResNet101": lambda: ResNet101(),
        "ResNet200": lambda: ResNet200(),
    }
    base_model_func = model_dict.get(args.model_type)
    if base_model_func is None:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model = SupConResNetFactory(base_model_func, feature_dim=128)

    if args.use_pretrained:
        checkpoint = torch.load(args.pretrained_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded pre-trained model weights.")
    else:
        print("Using randomly initialized ResNet.")
    model = model.to(device)

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32 if args.dataset_name != "imagenet" else 224, 32 if args.dataset_name != "imagenet" else 224).to(device)
        feature_dim = model.encoder(dummy_input).size(1)

    classifier = nn.Linear(feature_dim, num_classes).to(device)

    optimizer = optim.SGD(list(classifier.parameters()) + (list(model.parameters()) if not args.use_pretrained else []),
                          lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()

    print("Training started...")
    train_classifier(train_loader, val_loader, model, classifier, optimizer, scheduler, criterion,
                     device,epochs=args.epochs, save_dir=args.save_dir, model_type=args.model_type,
                     batch_size=args.batch_size, use_pretrained=args.use_pretrained,dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()



# python train_pretrained_classifier.py --model_type ResNet34 --batch_size 32 --epochs 30 --learning_rate 0.1 --dataset_name cifar100  --pretrained_model ./saved_models/pretraining/ResNet34/ResNet34_cifar10_feat128_supout_epoch241_batch32.pth


# python train_classifier.py --model_type ResNet34 --pretrained_model ./saved_models/pretraining/ResNet34/ResNet34_cifar10_feat128_supout_epoch241_batch32.pth --save_dir ./saved_models/classification --batch_size 32 --epochs 20 --learning_rate 0.001 --dataset_name cifar10 --dataset ./data --gpu 0

# python main.py --no_pretrained --model_type ResNet34 --dataset_name cifar10 --batch_size 32 --epochs 20 --learning_rate 0.001 --save_dir ./saved_models/classification/non_pretrained
