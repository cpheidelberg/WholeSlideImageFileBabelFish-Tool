import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm

# argument parsing:
import argparse
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--train_dir', type=str, required=True, help='path to the folder containing the training images, organized in subfolders by class')


'''
What this script does:

resnet50 training according to https://medium.com/@anglilian/image-classification-with-resnet-pytorch-1e48a4c33905 
'''

def resnet_validation(model, criterion, val_loader):
    val_loss = 0
    accuracy = 0

    for images, labels in iter(val_loader):
        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)

        equality = labels.data == probabilities.max(dim=1)[1]
        accuracy += equality.type(torch.FloatTensor).mean()

    return val_loss, accuracy


def train_resnet_model(model, optimizer, criterion, train_loader, val_loader, epochs=50):
    plot_training = []
    plot_validation = []

    for e in tqdm([i for i in range(epochs)]):
        model.train()
        running_loss = 0

        for images, labels in iter(train_loader):
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate performance of each epoch
        model.eval()

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            validation_loss, accuracy = resnet_validation(model, criterion, val_loader)

        train_loss = running_loss / len(train_loader)

        print(
            "Epoch: {}/{}.. ".format(e + 1, epochs),
            "Training Loss: {:.3f}.. ".format(train_loss),
            "Validation Loss: {:.3f}.. ".format(
                validation_loss / len(val_loader)
            ),
            "Validation Accuracy: {:.3f}".format(accuracy / len(val_loader)),
        )

        plot_training.append(train_loss)
        plot_validation.append(validation_loss / len(val_loader))

    return model, plot_training, plot_validation


def load_resnet_model(filepath, class_mapping):
    """
    Loads a checkpoint and rebuilds the model.

    Input:
    filepath(str): Relative path to model checkpoint
    """

    act_function = nn.Softmax(dim=1) # nn.LogSoftmax(dim=1)
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        num_classes = len(class_mapping)

        if "resnet50" in checkpoint["arch"]:
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            classifier = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(num_ftrs, num_classes)),
                        ("output", act_function),
                    ]
                )
            )
            model.fc = classifier

        elif "vgg16" in checkpoint["arch"]:
            '''model = models.vgg16(pretrained=True)
            num_ftrs = model.classifier[-1].out_features
            model.classifier.add_module("fc", nn.Linear(num_ftrs, num_classes))
            model.classifier.add_module("output", nn.LogSoftmax(dim=1))'''
            raise NotImplementedError("VGG16 not implemented yet.")

        else:
            return print("Architecture not recognized.")

        for param in model.parameters():
            param.requires_grad = False

        model.class_to_idx = checkpoint["class_to_idx"]
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    else:
        raise FileNotFoundError("Checkpoint file not found.")

def save_checkpoint(model, class_mapping, arch, store_dir="../checkpoint"):
    """
    Save trained model weights.

    Input:
    arch(str): Model architecture
    """

    checkpoint = {
        "arch": arch,
        "class_to_idx": class_mapping,
        "model_state_dict": model.state_dict(),
    }

    torch.save(checkpoint, f"{store_dir}/custom_{arch}.pth")

def main():
    ### params:
    resize = (395, 1155)
    relative_label_width = 0.3
    freeze = False
    epochs = 20
    run_name = "run4" + ("_freeze" if freeze else "")
    loss_function = nn.CrossEntropyLoss() # nn.NLLLoss()
    activ_function =  nn.Softmax(dim=1) #nn.LogSoftmax(dim=1) or nn.Softmax(dim=1), for crossentropyloss

    args = parser.parse_args()
    dataset_dir = args.train_dir

    cache_dir = f"resnet_cache/{run_name}"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    ### plot histogram of the labels
    img_data = {}

    for folder in os.listdir(dataset_dir):
        subfolder_path = dataset_dir + folder + '/'
        img_data[folder] = {}
        for subfolder in os.listdir(subfolder_path):
            img_data[folder][subfolder] = len(os.listdir(subfolder_path + subfolder))

    df = pd.DataFrame.from_dict(img_data, orient='index')
    df.T.plot(kind='bar')
    plt.savefig(f'{cache_dir}/label_histogram.png')
    plt.clf()
    plt.cla()
    plt.close()


    ### load data
    transformation = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.Lambda(lambda img: img.crop((0, 0, img.width * relative_label_width, img.height))),
            transforms.ToTensor()
        ])

    # Load data
    train_data = datasets.ImageFolder(
        dataset_dir + "train", transform=transformation
    )
    val_data = datasets.ImageFolder(
        dataset_dir + "val", transform=transformation
    )
    test_data = datasets.ImageFolder(
        dataset_dir + "test", transform=transformation
    )

    # Creating data samplers and loaders:
    BATCH_SIZE = 20
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False
    )

    # Label mapping
    class_mapping = train_data.class_to_idx

    ### load model
    model = models.resnet50(weights=True)

    if freeze:
        # Freeze pretrained model parameters to avoid backpropogating through them
        for parameter in model.parameters():
            parameter.requires_grad = False

    print("Original final layer")
    print(model.fc)

    # Build custom classifier
    num_classes = len(class_mapping)
    num_ftrs = model.fc.in_features
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc", nn.Linear(num_ftrs, num_classes)),
                ("output", activ_function),
            ]
        )
    )
    model.fc = classifier

    print("\nModified final layer")
    print(model.fc)

    # Loss function and gradient descent
    criterion = loss_function #
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Train model
    resnet_model, plot_train, plot_val = train_resnet_model(model, optimizer, criterion, train_loader, val_loader, epochs=epochs)
    plt.plot(range(len(plot_train)), plot_train, label='training')
    plt.plot(range(len(plot_val)), plot_val, label='validation')
    plt.legend()
    plt.savefig(f'{cache_dir}/training_validation_loss.png')

    save_checkpoint(resnet_model, class_mapping, 'resnet50', store_dir=cache_dir)

    # Evaluate model
    from sklearn.metrics import classification_report
    model.eval()
    with torch.no_grad():
        predictions = []
        true_labels = []

        for images, labels in iter(test_loader):
            output = model.forward(images)
            probabilities = torch.exp(output)
            predictions += probabilities.max(dim=1)[1].tolist()
            true_labels += labels.tolist()

    clf_report = str(classification_report(true_labels, predictions))

    print(clf_report)

    with open(f'{cache_dir}/classification_report.txt', 'w') as f:
        f.write(clf_report)

if __name__ == "__main__":
    main()