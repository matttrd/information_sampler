import pickle as pkl
import os
from IPython import embed
from torchvision import datasets, transforms
from loader import  load_data
from torch.utils.data import Dataset
import argparse
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, source, train, download, transform):
        if data == 'cifar10':
            self.data =  datasets.CIFAR10(source, train=train, download=download, transform=transform)
        elif data == 'cifar100':
            self.data =  datasets.CIFAR100(source, train=train, download=download, transform=transform)
        elif data == 'tinyimagenet64':
            source = source + 'tiny-imagenet-200/'
            ddir = source + 'train/' if train else source+'val/'
            self.data = datasets.ImageFolder(ddir, transform=transform)
        elif data == 'imagenet':
            source = source + 'imagenet/'
            ddir = source + 'train/' if train else source+'val/'
            self.data = datasets.ImageFolder(ddir, transform=transform)
        else:
            print('Only CIFAR 10/100 allowed!')

    def __getitem__(self, index):
        data, target = self.data[index][0], self.data[index][1]
        return data, target, index

    def __len__(self):
        return len(self.data)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
source =  '../data/'
train_dataset = MyDataset('cifar10', source, train=True, download=True,
                transform=transform_train)


def print_mask(path):
    # Loading the mask
    file = open(path, 'rb')
    corr_indeces = pkl.load(file)

    # Converting the mask from boolean to index
    actual_indeces = []
    if False in corr_indeces:
        for index in range(len(corr_indeces)):
            if corr_indeces[index] == True:
                actual_indeces.append(index)
        corr_indeces = np.array(actual_indeces)

    count = [0  for i in range(10)]
    for index in corr_indeces:
        count[train_dataset[index][1]] += 1

    print(count)
    return count

parser = argparse.ArgumentParser(description='check_masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp', default=f'..{os.sep}results', required=True)
opt = vars(parser.parse_args())

counts = []
exp_path = os.path.join(f'..{os.sep}results', opt['exp'])
for runs in os.listdir(exp_path):
    for file in os.listdir(os.path.join(exp_path, runs)):
        if file.split('_')[0] == 'indices':
            counts.append(print_mask(os.path.join(exp_path, runs, file)))
