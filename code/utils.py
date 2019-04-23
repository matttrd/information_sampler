import torch
import torch.utils.data
import torchvision


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
                