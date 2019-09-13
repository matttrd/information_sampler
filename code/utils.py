import torch
import torch.utils.data
import torchvision
from scipy.spatial.distance import euclidean 
import pandas as pd 

def get_label(self, dataset, idx):
    dataset_type = type(dataset)
    if dataset_type is torchvision.datasets.MNIST or \
            dataset_type is torchvision.datasets.CIFAR10:
                if isinstance(dataset.train_labels, torch.Tensor):
                    return dataset.train_labels[idx].item()
                else:
                    dataset.train_labels[idx]

    elif dataset_type is torchvision.datasets.ImageFolder:
        return dataset.imgs[idx][1]
    else:
        raise NotImplementedError

def get_imbalance_weights(dataset,indices=None, num_samples=None):

        # if indices is not provided, 
        # all elements in the dataset will be considered
    self.indices = list(range(len(dataset))) \
        if indices is None else indices
        
    # if num_samples is not provided, 
    # draw `len(indices)` samples in each iteration
    num_samples = len(indices) \
        if num_samples is None else num_samples
        
    # distribution of classes in the dataset 
    label_to_count = {}
    for idx in indices:
        label = get_label(dataset, idx)
        if label in label_to_count:
            label_to_count[label] += 1
        else:
            label_to_count[label] = 1
            
    # weight for each sample
    weights = [1.0 / label_to_count[get_label(dataset, idx)]
               for idx in self.indices]
    self.weights = torch.DoubleTensor(weights)

def get_clustering_indices_to_remove(weights_all_epochs, centroids, assignments, perc, mode, seed):
    # create dataframe
    weights_all_epochs = weights_all_epochs.tolist()
    df = pd.DataFrame()
    df['weights'] = weights_all_epochs
    df['cluster'] = assignments
    distances = []
    for index, row in df.iterrows():
        distances.append(euclidean(row.weights, centroids[row.cluster]))
    df['dist'] = distances
    idx = []
    # first pass through clusters
    second_pass = []
    for cluster in range(df['cluster'].nunique()):
        df_cluster = df[df['cluster']==cluster]
        if len(df_cluster) < 2:
            continue
        cl_popul_rate = len(df_cluster)/len(df)
        num_samples_to_drop = int(perc*len(df)*cl_popul_rate)
        if (perc*len(df)*cl_popul_rate - num_samples_to_drop > 0.5):
            second_pass.append(True)
        else:
            second_pass.append(False)
        if mode == 3:
            idx += list(df_cluster.nlargest(num_samples_to_drop, ['dist']).index.values)
        if mode == 4:
            idx += list(df_cluster.nsmallest(num_samples_to_drop, ['dist']).index.values)
        elif mode == 5:
            idx += list(df_cluster.sample(num_samples_to_drop, random_state=seed).index.values)
    df.drop(idx, inplace=True)
    # second pass through clusters: priority to clusters which deserved a "round up"
    for cluster in range(df['cluster'].nunique()):
        df_cluster = df[df['cluster']==cluster]
        if len(df_cluster) < 2 or (second_pass[cluster]==False):
            continue
        if mode == 3:
            idx += list(df_cluster.nlargest(1, ['dist']).index.values)
            df.drop(list(df_cluster.nlargest(1, ['dist']).index.values), inplace=True)
        if mode == 4:
            idx += list(df_cluster.nsmallest(1, ['dist']).index.values)
            df.drop(list(df_cluster.nsmallest(1, ['dist']).index.values), inplace=True)
        elif mode == 5:
            idx += list(df_cluster.sample(1, random_state=seed).index.values)
            df.drop(list(df_cluster.sample(1, random_state=seed).index.values), inplace=True)
        if perc*len(df) - len(idx) == 0:
            break
    # additional passes through clusters 
    while perc*len(df) - len(idx) > 0:
        for cluster in range(df['cluster'].nunique()):
            df_cluster = df[df['cluster']==cluster]
            if len(df_cluster) < 2:
                continue
            if mode == 3:
                idx += list(df_cluster.nlargest(1, ['dist']).index.values)
                df.drop(list(df_cluster.nlargest(1, ['dist']).index.values), inplace=True)
            if mode == 4:
                idx += list(df_cluster.nsmallest(1, ['dist']).index.values)
                df.drop(list(df_cluster.nsmallest(1, ['dist']).index.values), inplace=True)
            elif mode == 5:
                idx += list(df_cluster.sample(1, random_state=seed).index.values)
                df.drop(list(df_cluster.sample(1, random_state=seed).index.values), inplace=True)
            if perc*len(df) - len(idx) == 0:
                break
    return idx