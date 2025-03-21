import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns # todo: add to requirements
import os, sys
import pickle
import torch

class PCAAnalysis:
    def __init__(self, df, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.explained_variance_ratio = None
        if isinstance(df, pd.DataFrame):
            self.label = df['label']

            if 'embedding' in df.columns:
                self.data = df['embedding'].apply(pd.Series)
            else:
                self.data = df.drop(columns=['label'])

        if isinstance(df, dict):
            self.label = df['label']
            self.data = df['data']

    def fit(self):
        X_scaled = self.scaler.fit_transform(self.data)
        self.pca.fit(X_scaled)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.df_pca = pd.DataFrame(self.pca.fit_transform(X_scaled),
                                   columns=[f'PC{i+1}' for i in range(self.n_components)])
        self.df_pca['label'] = self.label.to_list()

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        return self.pca.fit_transform(X_scaled)

    def plot_explained_variance(self, save_path = None, title = None):

        if self.explained_variance_ratio is None:
            self.fit()

        plt.close()
        #self.df_pca.label = [str(i) for i in self.df_pca.label]
        sns.scatterplot(x=self.df_pca["PC1"],
                   y=self.df_pca["PC2"],
                   hue=self.df_pca.label,
                   palette="tab10", legend="full")

        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        if title is not None:
            plt.title(title)
        else:
            title = os.path.basename(save_path)
            plt.title(str(title).replace('.png', ''))

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    def get_components(self):
        """
        Get the principal components (eigenvectors).
        :return: Principal components
        """
        return self.pca.components_

    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio.
        :return: Explained variance ratio
        """
        return self.explained_variance_ratio

def dump_features(df, save_path):

    features= df['embedding'].tolist()
    features = torch.stack(features)
    label = df['class_name'].tolist()
    if "set" in df.columns:
        setdata = df['set'].tolist()

    with open(save_path, "wb") as f:
        if not "set" in df.columns:
            pickle.dump({"features": features, "label": label}, f)
        else:
            pickle.dump({"features": features,
                         "label": label,
                         "set": setdata}, f)
    print(f"features (and label) are saved to {save_path}")

def load_features(save_path):

    with open(save_path, "rb") as f:
        data = pickle.load(f)
    print(f"features (and label) loaded from {save_path}")

    # Access the tensor and list
    features = data["features"]
    features = features.tolist()
    features = [torch.tensor(i) for i in features]
    label= data["label"]

    if "set" in data.keys():
        setdata = data['set']

    if not "set" in data.keys():
        df = pd.DataFrame({'class_name': label, 'embedding': features})
    else:
        df = pd.DataFrame({'class_name': label,
                           'embedding': features,
                           "set": setdata})

    return df
