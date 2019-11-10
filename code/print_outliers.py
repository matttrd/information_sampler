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

# Other possible way to print images:
# https://corochann.com/cifar-10-cifar-100-dataset-introduction-1258.html


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

def print_from_ordered_vector(run_path, index,  exp_type, count_type, less_counts, size_plot):
    images, labels = [], []
    if less_counts == 1:
        for i in range(size_plot):
            images.append(train_dataset[index[int(i)]][0])
            labels.append(train_dataset[index[int(i)]][1])
    else:
        for i in range(size_plot):
            images.append(train_dataset[index[int(-i-1)]][0])
            labels.append(train_dataset[index[int(-i-1)]][1])

    saving_path = os.path.join(run_path, 'analysis_exctracted_images')
    os.makedirs(saving_path, exist_ok=True)
    if less_counts == 1:
        exp_type += '_less_' + count_type + str(size_plot)
    else:
        exp_type += '_more_' + count_type + str(size_plot)
    saving_path = os.path.join(saving_path, exp_type + '.pdf')
    image_batch = torch.tensor(np.stack(images))
    # image_batch = torch.tensor(np.stack(train_dataset[18310][0]), train_dataset[18310][1]) Cifar10 Easter egg!
    imsave(torchvision.utils.make_grid(image_batch), saving_path, labels)


def print_spotted_outliers(run_path, remaining_path, exp_type, less_counts=1, size_plot=104):
    path = os.path.join(run_path, remaining_path)
    f = open(path, 'rb')
    counts = pkl.load(f)
    index = np.argsort(counts)        # from small to large
    # sorted_counts = counts[index]

    print_from_ordered_vector(run_path, index, exp_type, 'counted_images_', less_counts, size_plot)


def print_count_based_outliers(exp_path, less_counts=1, ith_image=[0]):
    if exp_path.split(f'{os.sep}')[-1] == 'analysis_experiments':
        return # This folder is not a run!!!
    for file in os.listdir(os.path.join(exp_path, 'sample_counts')):
        epoch = int(file.split('_')[-1].split('.')[0])
        remaining_path = os.path.join('sample_counts', file)
        if epoch in opt['epochs']: #Get epoch from the saved counter file
            print('Processing Image: ' + str(ith_image[0]))
            print_spotted_outliers(exp_path, remaining_path, 'count_based_' + str(epoch) , less_counts=less_counts, size_plot=104)
            ith_image[0] += 1

from analyze_forgetting_stats import compute_forgetting_statistics, sort_examples_by_forgetting
from count_visualization import get_params_from_log, kde_sklearn
def print_forgetting_based_outliers(run_path, less_counts=1, size_plot=104):
    # embed()
    if run_path.split(f'{os.sep}')[-1] == 'analysis_experiments':
        return # This folder is not a run!!!
    f = open(os.path.join(run_path, 'forgetting_stats', 'stats.pkl'), 'rb')
    loaded = pkl.load(f)
    # Compute the forgetting statistics per example for training run
    _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(
        loaded, 200) # note 200 is the max number of epochs
    # Sort examples by forgetting counts in ascending order, over one or more training runs
    # from never forgotten to never learnt
    # orderred_examples are the indeces of the dataset ordered
    # ordered_values are the number of forgetting events
    index, ordered_values = sort_examples_by_forgetting(
        unlearned_per_presentation, first_learned, 200) # note 200 is the max number of epochs

    # Plot kde
    plt.figure()
    info_exp = get_params_from_log(os.path.join(run_path, 'logs', 'flogger.log'))
    save_path = os.path.join(run_path, 'analysis')
    os.makedirs(save_path, exist_ok=True)
    title = info_exp['dataset'] + ' ' + info_exp['arch'] + ' BS' + str(info_exp['b']) + ' ' + info_exp['sampler'] + ' norm ' + str(info_exp['normalizer']) + ' Temp' + str(info_exp['temperature'])
    plt.title(title)
    x_grid = np.linspace(0, max(ordered_values), max(ordered_values))
    pdf = kde_sklearn(ordered_values, x_grid, bandwidth=0.5)
    plt.plot(x_grid, pdf, label="Mean: " + str(np.mean(np.array(ordered_values))) + " std: " + str(np.std(np.array(ordered_values))), linewidth=2)
    plt.grid(), plt.legend(), plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'forgetting_kde.pdf'), dpi=2560, bbox_inches='tight')
    plt.close()

    # Plot hist
    plt.figure(), plt.title(title)
    plt.hist(ordered_values, bins=210, label="Mean: " + str(np.mean(np.array(ordered_values))) + " std: " + str(np.std(np.array(ordered_values))), linewidth=2)
    plt.grid(), plt.legend(), plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'forgetting_hist.pdf'), dpi=2560, bbox_inches='tight')
    plt.close()

    print_from_ordered_vector(run_path, index, 'forget_based', 'forgotten_images_', less_counts, size_plot)



parser = argparse.ArgumentParser(description='check_masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sd', default=f'..{os.sep}results') #Note this is not used now since we are saving on the exp run folder
parser.add_argument('--exp', default=f'..{os.sep}results', required=True)
parser.add_argument('--epochs', default=[0,59,119,159])
opt = vars(parser.parse_args())

dataset = opt['exp'].split('_')[-1]


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

source =  '../data/'

if dataset == 'cifar10':
    CLASSES = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

elif dataset == 'cifar100':
    CLASSES = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
    )
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
else:
    raise NotImplementedError

train_dataset = MyDataset(dataset, source, train=True, download=True,
                transform=transform_train)


exp_path = os.path.join(f'..{os.sep}results', opt['exp'])

for folder in os.listdir(exp_path):
    path = os.path.join(exp_path, folder)
    print_count_based_outliers(path, less_counts=1)
    print_count_based_outliers(path, less_counts=-1)
    try:
        print_forgetting_based_outliers(path, less_counts=1)
        print_forgetting_based_outliers(path, less_counts=-1)
    except FileNotFoundError:
        print('Run: ' + path + ' has not any forgetting stats saved.')
