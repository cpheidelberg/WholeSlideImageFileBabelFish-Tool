import os, sys
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet152, resnet50
from tqdm import tqdm
from torchvision.datasets import VisionDataset
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn as sns
from few_shot_utils import PCAAnalysis
from easyfsl.datasets import FeaturesDataset
from few_shot_utils import dump_features, load_features

# argument parsing:
import argparse
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--train_dir', type=str, required=True, help='path to the folder containing the training images, organized in subfolders by class')



'''class PrototypicalNetworks(nn.Module):
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
        return scores'''

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

########## functions ###############

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main():

    classes_to_include = ["IHC", "SlideMateLaser", "CMCP", "SuperFrost", "DARK"] # "SuperFrost"
    relative_label_width = 0.3 # to crop the labels from the macro-images
    plot_example_macro_images = True
    resize = (395, 1155)
    batch_size = 8
    plot_variance = True
    num_epochs = 20  # Number of episodes
    backbone_name = "resnet_cache/run4/custom_resnet50.pth" # one of ["resnet18", "resnet34", "resnet50", "resnet152", or any .pth file]
    with_wandb = False
    val_instances_per_class = 10
    lr = 0.001
    grayscale_images = False
    update_embedding = True
    N_SHOT_TRAIN = 10  # Number of images per class in the support set
    N_QUERY_TRAIN = 10  # Number of images per class in the query set
    N_SHOT_VAL = 5
    N_QUERY_VAL = 5
    N_TASKS_TRAIN = 100
    N_TASKS_VAL = 1
    fig_dpis = 100

    args = parser.parse_args()
    dataset_dir = args.train_dir

    if grayscale_images:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.Lambda(lambda img: img.crop((0, 0, img.width*relative_label_width, img.height))),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.Lambda(lambda img: img.crop((0, 0, img.width*relative_label_width, img.height))),
            transforms.ToTensor()
        ])

    # load dataset and split it into training and test set
    train_ds = FolderStructuredMacroDataset(root=dataset_dir, transform=transform, classes_to_include=classes_to_include)
    val_ds = FolderStructuredMacroDataset(root=dataset_dir, transform=transform, classes_to_include=classes_to_include)
    val_ds.samples = []
    for class_name in val_ds.classes:
        instances_of_class = [inst for inst in train_ds.samples if inst[1] == val_ds.class_to_idx[class_name]]
        assert len(instances_of_class) > val_instances_per_class, f"Class {class_name} has less than {val_instances_per_class} instances in the training set ({len(instances_of_class)})."
        instances_to_add_to_test = instances_of_class[:val_instances_per_class]
        val_ds.samples = val_ds.samples + instances_to_add_to_test
    # remove all instances from the training set that are now in the test set
    train_ds.samples = [inst for inst in train_ds.samples if inst not in val_ds.samples]

    # define get_labels function:
    val_ds.get_labels = lambda: [instance[1] for instance in val_ds.samples]
    train_ds.get_labels = lambda: [instance[1] for instance in train_ds.samples]

    # print some dataset statistics
    dataset_name = '_'.join([f"{train_ds.class_to_idx[class_name]}{class_name}" for class_name in train_ds.classes])
    print(f"=== Dataset {dataset_name} Stats ===")
    print(f"classes: {train_ds.class_to_idx}")
    print(f"class histogram train_set:")
    for class_name in train_ds.classes:
        print(
            f"\t{class_name}: {len([inst for inst in train_ds.samples if inst[1] == train_ds.class_to_idx[class_name]])}")
    print(f"class histogram val_set:")
    for class_name in val_ds.classes:
        print(
            f"\t{class_name}: {len([inst for inst in val_ds.samples if inst[1] == val_ds.class_to_idx[class_name]])}")
    print()

    # create few-shot_cache folder if not exists
    if not os.path.exists("few-shot_cache"):
        os.mkdir("few-shot_cache")

    # generate a dataframe as dataset and store it as table
    df_dataset = pd.DataFrame(columns=["ID", "diagnosis_id", "diagnosis", "file_path", "set"])
    for i, (img_path, label) in enumerate(train_ds.samples):
        df_dataset.loc[i] = [i, label, train_ds.classes[label], img_path, "train"]
    for i, (img_path, label) in enumerate(val_ds.samples):
        df_dataset.loc[i + len(train_ds.samples)] = [i, label, val_ds.classes[label], img_path, "val"]
    df_dataset.to_csv("last_processed_dataset.csv", index=False)

    # load a pretrained resnet as model
    if backbone_name == "resnet18":
        resnet = resnet18(pretrained=True)
    elif backbone_name == "resnet34":
        resnet = resnet34(pretrained=True)
    elif backbone_name == "resnet152":
        resnet = resnet152(pretrained=True) # model will be downloaded if not present in environment
    elif backbone_name == "resnet50":
        resnet = resnet50(pretrained=True)
    elif '.pth' in backbone_name:
        from resnet_pretraining import load_resnet_model
        print(f"loading resnet model from {backbone_name}")
        resnet = load_resnet_model(backbone_name, train_ds.class_to_idx)
        backbone_name = os.path.basename(backbone_name).replace(".pth", "")
    else:
        raise ValueError(f"backbone_name {backbone_name} not supported. Use one of ['resnet18', 'resnet34', 'resnet152']")
    resnet.fc = nn.Flatten()


    #### render preview of support and query images
    if plot_example_macro_images:

        # plot a figure which shows 5 example images for each class:
        plt.clf()
        plt.figure(figsize=(20, 20))
        for i, class_name in enumerate(train_ds.classes):
            instances_of_class = [inst for inst in train_ds.samples if inst[1] == train_ds.class_to_idx[class_name]]
            plt.subplot(1, len(train_ds.classes), i + 1)
            plt.title(f"{train_ds.class_to_idx[class_name]}: {class_name}")
            plt.axis("off")
            for j, (img_path, label) in enumerate(instances_of_class[:5]):
                img = Image.open(img_path)
                plt.imshow(img)
                plt.axis("off")
        plt.savefig(f"few-shot_cache/IMG#{dataset_name}.png", dpi=fig_dpis)

        # The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
        val_ds.get_labels = lambda: [
            instance[1] for instance in val_ds.samples
        ]
        test_sampler = TaskSampler(
            val_ds, n_way=len(val_ds.classes), n_shot=N_SHOT_VAL, n_query=N_QUERY_VAL, n_tasks=N_TASKS_VAL
        )
        test_loader = DataLoader(
            val_ds,
            batch_sampler=test_sampler,
            num_workers=12,
            pin_memory=True,
            collate_fn=test_sampler.episodic_collate_fn
        )
        (
            example_support_images,
            example_support_labels,
            example_query_images,
            example_query_labels,
            example_class_ids,
        ) = next(iter(test_loader))
        plt.clf()
        plot_images(example_support_images, "transformed images", images_per_row=5)
        plt.savefig(f"few-shot_cache/IMGT#{dataset_name}.png", dpi=fig_dpis)


    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size, num_workers=0)
    for xb, yb in train_loader:
      print(f"xb.shape: {xb.shape}")
      print(f"Normalization-test: min value is {torch.min(xb)} and max value is {torch.max(xb)}")
      print(f"yb.shape: {yb.shape}")
      break

    resnet_out = resnet(xb)

    device = get_default_device()

    ### predict or load (labeled) backbone-embeddings
    from easyfsl.utils import predict_embeddings
    embedding_backup_name = "few-shot_cache/embeddings_" + dataset_name + "_" + backbone_name +".pkl"
    # if not file exists, create it
    if not os.path.exists(embedding_backup_name) or update_embedding:
        if not update_embedding:
            print(f"file {embedding_backup_name} does not exist. "
                  f"precomputing embeddings now. Will store them in {embedding_backup_name}")
        embeddings_df_train = predict_embeddings(train_loader, resnet.to(device), device=device)
        embeddings_df_train['set'] = "train"
        embeddings_df_val = predict_embeddings(val_loader, resnet.to(device), device=device)
        embeddings_df_val['set'] = "val"
        embeddings_df = pd.concat((embeddings_df_train, embeddings_df_val))
        dump_features(embeddings_df, embedding_backup_name)
    else:
        embeddings_df = load_features(embedding_backup_name)
        print(f"loaded embeddings from {embedding_backup_name}")

    if plot_variance:
        # plot variance in latent space with PCA:
        df = embeddings_df
        df = df.rename(columns={'class_name': 'label'})
        l_space = PCAAnalysis(df)

        l_space.plot_explained_variance(save_path=f"few-shot_cache/PCA#{dataset_name}_{backbone_name}.png")

    # create a dataset from the embeddings
    embeddings_df_train = embeddings_df[embeddings_df['set'] == 'train'].reset_index()
    embeddings_df_val = embeddings_df[embeddings_df['set'] == 'val'].reset_index()
    features_dataset_train = FeaturesDataset.from_dataframe(embeddings_df_train)
    features_dataset_val = FeaturesDataset.from_dataframe(embeddings_df_val)
    #print(f"example feature: {features_dataset_train[0]}")

    # create a dataloader and stuff from the dataset
    class_ids = np.unique(embeddings_df.class_name)
    task_sampler_train = TaskSampler(
        features_dataset_train,
        n_way=len(class_ids),
        n_shot=N_SHOT_TRAIN,
        n_query=N_QUERY_TRAIN,
        n_tasks=N_TASKS_TRAIN,
    )
    features_loader_train = DataLoader(
        features_dataset_train,
        batch_sampler=task_sampler_train,
        num_workers=0,
        pin_memory=True,
        collate_fn=task_sampler_train.episodic_collate_fn,
    )
    task_sampler_val = TaskSampler(
        features_dataset_val,
        n_way=len(class_ids),
        n_shot=N_SHOT_VAL,
        n_query=N_QUERY_VAL,
        n_tasks=N_TASKS_VAL,
    )
    features_loader_val = DataLoader(
        features_dataset_val,
        batch_sampler=task_sampler_val,
        num_workers=0,
        pin_memory=True,
        collate_fn=task_sampler_val.episodic_collate_fn,
    )

    # create prototypical network as model
    from easyfsl.methods import PrototypicalNetworks
    model = PrototypicalNetworks(backbone=nn.Linear(
        features_dataset_train[0][0].shape[0],
        len(embeddings_df.class_name.unique())))

    ### first lets try some shots with the prototypical network without any training:
    support_set = torch.stack(embeddings_df_train.embedding.tolist())
    support_label = torch.tensor(embeddings_df_train.class_name.tolist())
    model.process_support_set(
        support_set.to(device), support_label.to(device)
    )

    y_true, y_pred = [], []
    with tqdm(total=len(val_loader), file=sys.stdout) as pbar:
        for i in range(len(embeddings_df_val)):
            val_features = embeddings_df_val.embedding[i]

            '''model.process_support_set(
                support_set.to(device), support_label.to(device)
            )'''

            out = model(val_features.to(device).unsqueeze(0))

            y_pred.append(int(torch.argmax(out).detach().cpu().numpy()))
            y_true.append(int(embeddings_df_val.class_name[i]))

            pbar.set_description('validation independent from training and feature generation')
            pbar.update(1)

    kappa_without_finetuning = cohen_kappa_score(y_true, y_pred)
    print(f"Kappa without finetuning: {kappa_without_finetuning}. (using train-set as support and val as query)")

    # prepare weights and biases session
    if with_wandb:
        import wandb # todo: add this to requirements OR try to remove wandb usage!
        wandb.init(project=dataset_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    ################# training loop #################
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for support_set, support_label, query_set, querry_label, label_included in features_loader_train:
            optimizer.zero_grad()

            # Forward pass through ProtoNet
            model.process_support_set(
                support_set.to(device), support_label.to(device)
            )
            out = model(query_set.to(device))

            # Backpropagation
            loss = criterion(out, querry_label.to(device))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / len(querry_label)

        y_pred, y_true = [], []
        epoch_loss_val = 0
        for support_set, support_label, query_set, querry_label, label_included in features_loader_val:
            # Forward pass through ProtoNet
            model.process_support_set(
                support_set.to(device), support_label.to(device)
            )
            out = model(query_set.to(device))
            loss = criterion(out, querry_label.to(device))
            y_pred.append(torch.argmax(out, dim=1).detach().cpu().numpy())
            y_true.append(querry_label.cpu().numpy())
            epoch_loss_val += loss.item() / len(querry_label)

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        kappa_after_finetuning = cohen_kappa_score(y_true, y_pred)
        #cm = confusion_matrix(y_true, y_pred)
        # if kappa > 0.9: break
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f} & {epoch_loss_val:.4f}, kappa: {kappa_after_finetuning:.4f}")

        # Log losses to Weights & Biases
        if with_wandb:
            wandb.log({"Train Loss": epoch_loss,
                       "Validation Loss": epoch_loss_val,
                       "Validation Kappa": kappa_after_finetuning})

    if with_wandb:
        wandb.finish()

    #plot confusion matrix
    '''import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues"
                )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (for validation set)")
    plt.show()'''

    # final validation + confusion matrix
    #model = model.to(device) # reset model?
    loss_val, score_val = [], []

    support_set = torch.stack(embeddings_df_train.embedding.tolist())
    support_label = torch.tensor(embeddings_df_train.class_name.tolist())
    model.process_support_set(
        support_set.to(device), support_label.to(device)
    )

    y_true, y_pred = [], []
    with tqdm(total=len(val_loader), file=sys.stdout) as pbar:
        for i in range(len(embeddings_df_val)):
            val_features = embeddings_df_val.embedding[i]

            '''model.process_support_set(
                support_set.to(device), support_label.to(device)
            )'''
            out = model(val_features.to(device).unsqueeze(0))

            y_pred.append(int(torch.argmax(out).detach().cpu().numpy()))
            y_true.append(int(embeddings_df_val.class_name[i]))

            pbar.set_description('validation independent from training and feature generation')
            pbar.update(1)

    kappa_after_finetuning = cohen_kappa_score(y_true, y_pred)
    print(f"Kappa value after finetuning: {kappa_after_finetuning} (using train-set as support and val as query)")
    print(f"Kappa improvement: +{kappa_after_finetuning - kappa_without_finetuning}")

    # todo: store finetuned Protonet model.

    # plot confusion matrix
    # clear the current figure
    plt.clf()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (for validation set)")
    plt.savefig(f"few-shot_cache/CM#{dataset_name}_{backbone_name}.png", dpi=fig_dpis)

if __name__ == "__main__":
    main()