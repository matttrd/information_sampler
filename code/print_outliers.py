import pickle as pkl
import os
from IPython import embed
from torchvision import datasets, transforms
from loader import  load_data
from torch.utils.data import Dataset
import torch
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt

# If CIFAR10
CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

def imsave(img, sav_path, labels):
    # Prepare the labels as a formatted string
    string_labels = ''
    for index, lab in enumerate(labels):
        if (index % 8) == 0:
            string_labels += '\n\n\n '
        string_labels += str(CLASSES[lab]) + '     '

    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(-400, 420, string_labels, fontsize=6, ha='left', wrap=True)
    plt.savefig(sav_path, dpi=2560, bbox_inches='tight')
    plt.close()

def print_spotted_outliers(run_path, remaining_path, exp_type, less_counts=1, size_plot=104):
    path = os.path.join(run_path, remaining_path)
    f = open(path, 'rb')
    counts = pkl.load(f)
    index = np.argsort(counts)        # from small to large
    # sorted_counts = counts[index]
    images, labels = [], []
    for i in range(size_plot):
        images.append(train_dataset[index[int(less_counts * i)]][0])
        labels.append(train_dataset[index[int(less_counts * i)]][1])

    saving_path = os.path.join(run_path, 'analysis_exctracted_images')
    os.makedirs(saving_path, exist_ok=True)
    if less_counts == 1:
        exp_type += '_less_counted_images_' + str(size_plot)
    else:
        exp_type += '_more_counted_images_' + str(size_plot)
    saving_path = os.path.join(saving_path, exp_type + '.pdf')
    image_batch = torch.tensor(np.stack(images))
    # image_batch = torch.tensor(np.stack(train_dataset[18310][0]), train_dataset[18310][1]) Cifar10 Easter egg!
    imsave(torchvision.utils.make_grid(image_batch), saving_path, labels)
    # embed()

def print_count_based_outliers(exp_path, less_counts=1, ith_image=[0]):
    for file in os.listdir(os.path.join(exp_path, 'sample_counts')):
        epoch = int(file.split('_')[-1].split('.')[0])
        remaining_path = os.path.join('sample_counts', file)
        if epoch in opt['epochs']: #Get epoch from the saved counter file
            print('Processing Image: ' + str(ith_image[0]) + ' out of ' + str( 2 * len(opt['epochs']) * len(os.listdir(os.path.join(exp_path, 'sample_counts')))))
            print_spotted_outliers(exp_path, remaining_path, 'count_based_' + str(epoch) , less_counts=less_counts, size_plot=104)
            ith_image[0] += 1

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
source =  '../data/'
train_dataset = MyDataset('cifar10', source, train=True, download=True,
                transform=transform_train)


parser = argparse.ArgumentParser(description='check_masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sd', default=f'..{os.sep}results') #Note this is not used now since we are saving on the exp run folder
parser.add_argument('--exp', default=f'..{os.sep}results', required=True)
parser.add_argument('--epochs', default=[0,59,119,159])
opt = vars(parser.parse_args())

exp_path = os.path.join(f'..{os.sep}results', opt['exp'])

for folder in os.listdir(exp_path):
    path = os.path.join(exp_path, folder)
    print_count_based_outliers(path, 1)
    print_count_based_outliers(path, -1)
