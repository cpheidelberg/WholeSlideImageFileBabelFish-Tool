import os, sys

import matplotlib.pyplot as plt
import torch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet152
from tqdm import tqdm
from torchvision.datasets import Omniglot, VisionDataset
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average
import numpy as np
import cv2

# argument parsing:
import argparse
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--train_dir', type=str, required=True, help='path to the folder containing the training images, organized in subfolders by class')



class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores

from PIL import Image
class FolderStructuredMacroDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, file_type=".png", classes_to_include=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if classes_to_include:
            assert all([cls_name in sorted(os.listdir(root)) for cls_name in classes_to_include]), f"Classes to include contain classes that are not in the dataset: {classes_to_include}"
            self.classes = sorted(classes_to_include)
        else:
            self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.file_type = file_type

        self.samples = self._load_samples()
        self.img_type = "RGB"

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.root, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(self.file_type):
                        img_path = os.path.join(class_path, img_name)
                        label = self.class_to_idx[class_name]
                        samples.append((img_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert(self.img_type)

        #macro_img_array = np.array(image)
        #image = cv2.equalizeHist(macro_img_array)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def main():

    classes_to_include = ["IHC", "SlideMateLaser", "SuperFrost"]
    relative_label_width = 0.3
    plot_query_support_images = False
    resize = (395, 1155)

    args = parser.parse_args()

    # load a custom image dataset as torchvision.datasets.VisionDataset object
    dataset_dir = args.train_dir
    #test_dir = args.test_dir

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.Lambda(lambda img: img.crop((0, 0, img.width*relative_label_width, img.height))),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # load dataset and split it into training and test set
    train_set = FolderStructuredMacroDataset(root=dataset_dir, transform=transform, classes_to_include=classes_to_include)
    test_set = FolderStructuredMacroDataset(root=dataset_dir, transform=transform, classes_to_include=classes_to_include)
    test_set.samples = []
    test_instances_per_class = 20
    for class_name in test_set.classes:
        instances_of_class = [inst for inst in train_set.samples if inst[1] == test_set.class_to_idx[class_name]]
        assert len(instances_of_class) > test_instances_per_class, f"Class {class_name} has less than {test_instances_per_class} instances in the training set ({len(instances_of_class)})."
        instances_to_add_to_test = instances_of_class[:test_instances_per_class]
        test_set.samples = test_set.samples + instances_to_add_to_test
    # remove all instances from the training set that are now in the test set
    train_set.samples = [inst for inst in train_set.samples if inst not in test_set.samples]
    print(f"class histogram train_set:")
    for class_name in train_set.classes:
        print(
            f"\t{class_name}: {len([inst for inst in train_set.samples if inst[1] == train_set.class_to_idx[class_name]])}")
    print(f"class histogram test_set:")
    for class_name in test_set.classes:
        print(
            f"\t{class_name}: {len([inst for inst in test_set.samples if inst[1] == test_set.class_to_idx[class_name]])}")

    # generate a dataframe as dataset and store it as table
    import pandas as pd
    df_dataset = pd.DataFrame(columns=["ID", "diagnosis_id", "diagnosis", "file_path", "set"])
    for i, (img_path, label) in enumerate(train_set.samples):
        df_dataset.loc[i] = [i, label, train_set.classes[label], img_path, "train"]
    for i, (img_path, label) in enumerate(test_set.samples):
        df_dataset.loc[i + len(train_set.samples)] = [i, label, test_set.classes[label], img_path, "val"]
    df_dataset.to_csv("last_processed_dataset.csv", index=False)

    # load a pretrained resnet as model
    convolutional_network = resnet152(pretrained=True) # model will be downloaded if not present in environment
    convolutional_network.fc = nn.Flatten()

    model = PrototypicalNetworks(convolutional_network)

    N_WAY = 3  # Number of classes in a task
    N_SHOT = 5  # Number of images per class in the support set
    N_QUERY = 10  # Number of images per class in the query set
    N_EVALUATION_TASKS = 100

    # The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
    test_set.get_labels = lambda: [
        instance[1] for instance in test_set.samples
    ]
    test_sampler = TaskSampler(
        test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
    )

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    #### render preview of support and query images
    if plot_query_support_images:
        (
            example_support_images,
            example_support_labels,
            example_query_images,
            example_query_labels,
            example_class_ids,
        ) = next(iter(test_loader))
        plot_images(example_support_images, "support images", images_per_row=N_SHOT)
        plot_images(example_query_images, "query images", images_per_row=N_QUERY)
        plt.show()



if __name__ == "__main__":
    main()

