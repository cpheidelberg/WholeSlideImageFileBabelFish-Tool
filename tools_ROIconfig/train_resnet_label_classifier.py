import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def get_data_loaders(data_dir, batch_size=32, test_size=0.2, val_size=0.1, random_seed=42):
    relative_label_width = 0.31
    transform = transforms.Compose([
        transforms.Resize((395, 1155)),
        transforms.Lambda(lambda img: img.crop((0, 0, int(img.width * relative_label_width), img.height))),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # use the ImageNet mean and std
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    # store an example image:
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    example_indices = [np.random.randint(len(dataset))]
    for i, ax in enumerate(axes.flatten()):
        img, label = dataset[example_indices[-1]]
        # Tensor umwandeln für Matplotlib
        img = img.numpy().transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalisierung
        img = np.clip(img, 0, 1)  # Werte begrenzen
        ax.imshow(img)
        ax.axis('off')
        random = np.random.randint(len(dataset))
        while random in example_indices:
            random = np.random.randint(len(dataset))
        example_indices.append(random)
    plt.tight_layout()
    plt.savefig('example_images_grid.png')
    plt.close()
    #print(f"stored example images in example_images_grid.png, picked indices: {example_indices}")

    train_indices, test_indices = train_test_split(indices, test_size=test_size, stratify=targets,
                                                   random_state=random_seed)
    train_targets = targets[train_indices]
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size / (1 - test_size),
                                                  stratify=train_targets, random_state=random_seed)

    # check that there are no intersections between the sets:
    assert len(set(train_indices) & set(val_indices)) == 0, "Train and validation indices overlap"
    assert len(set(train_indices) & set(test_indices)) == 0, "Train and test indices overlap"
    assert len(set(val_indices) & set(test_indices)) == 0, "Validation and test indices overlap"
    assert len(train_indices) == len(list(set(train_indices))), "Train indices contain duplicates"
    assert len(val_indices) == len(list(set(val_indices))), "Validation indices contain duplicates"
    assert len(test_indices) == len(list(set(test_indices))), "Test indices contain duplicates"

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset.classes


def get_resnet_model(model_name, num_classes):
    model = getattr(models, model_name)(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")


    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()

    torch.save(model.state_dict(), 'trained_model.pth')

    return model


def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    df = pd.DataFrame(report).transpose()
    df.to_csv('classification_report.csv', index=True)
    print(report_str)
    with open('classification_report.txt', 'w') as f:
        f.write(report_str)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # store the confusion matrix:
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv('confusion_matrix.csv')

    # store as image:
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add numbers into the cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2. else 'black')

    plt.savefig('confusion_matrix.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18',
                        help='ResNet model variant')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, class_names = get_data_loaders(args.data_dir, args.batch_size)
    model = get_resnet_model(args.model, len(class_names)).to(device)
    model = train_model(model, train_loader, val_loader, device, args.epochs, args.lr)
    evaluate_model(model, test_loader, device, class_names)


if __name__ == '__main__':
    main()
