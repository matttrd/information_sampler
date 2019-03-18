import numpy as np
import random
import argparse
import copy
import pickle as pkl
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Imbalanced dataset creator')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')                        
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


# original dataset
original_train_dataset = datasets.MNIST('./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            ]))

original_train_loader = torch.utils.data.DataLoader(
                            original_train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('./data', train=False, 
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])),
                    batch_size=args.test_batch_size, 
                    shuffle=True, **kwargs)

# imbalanced dataset
torch.manual_seed(args.seed)
num_classes = len(original_train_dataset.train_labels.unique())
class_labels = range(num_classes)
sample_probs = torch.rand(num_classes)
idx_to_del = [i for i, label in enumerate(original_train_loader.dataset.train_labels) 
              if random.random() > sample_probs[label]]
imbalanced_train_dataset = copy.deepcopy(original_train_dataset)
imbalanced_train_dataset.train_labels = np.delete(original_train_loader.dataset.train_labels, idx_to_del, axis=0)
imbalanced_train_dataset.train_data = np.delete(original_train_loader.dataset.train_data, idx_to_del, axis=0)

data = {'imbalanced_train_dataset': imbalanced_train_dataset, 'test_loader': test_loader, 'num_classes': num_classes}
with open('./imbalanced_dataset.pickle', 'wb') as handle:
    pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

# plot labels distribution (train) - original dataset
fig, ax = plt.subplots()
_, counts = np.unique(original_train_dataset.train_labels, return_counts=True)
ax.bar(class_labels, counts)
ax.set_xticks(class_labels)
plt.title('Labels distribution (train) - original dataset')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('./train_labels_distrib_original.png')
# plot labels distribution (train) - imbalanced dataset
fig, ax = plt.subplots()
_, counts = np.unique(imbalanced_train_dataset.train_labels, return_counts=True)
ax.bar(class_labels, counts)
ax.set_xticks(class_labels)
plt.title('Labels distribution (train) - imbalanced dataset')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('./train_labels_distrib_imbalanced.png')
# plot labels distribution (test)
fig, ax = plt.subplots()
_, counts = np.unique(test_loader.dataset.test_labels, return_counts=True)
ax.bar(class_labels, counts)
ax.set_xticks(class_labels)
plt.title('Labels distribution (test)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('./test_labels_distrib.png')