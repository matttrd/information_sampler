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
    return count, corr_indeces

parser = argparse.ArgumentParser(description='check_masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp', default=f'..{os.sep}results', required=True)
opt = vars(parser.parse_args())

counts = []
corr_indeces = []
exp_path = os.path.join(f'..{os.sep}results', opt['exp'])
for runs in os.listdir(exp_path):
    for file in os.listdir(os.path.join(exp_path, runs)):
        if file.split('_')[0] == 'indices':
            count, corr_ind = print_mask(os.path.join(exp_path, runs, file))
            counts.append(count)
            corr_indeces.append(corr_ind)

import matplotlib.pyplot as plt
def print_perc(save_path, info, IS_counts, corr_indeces):
    os.makedirs(save_path, exist_ok=True)
    epoch_filer = [0, 61, 121, 179]
    title = info['sampler'] + '_' + 'Temp' + '_' + str(info['temperature']).replace('.', '_') + '_' + 'norm_' + str(info['norm'])
    for epoch in IS_counts.keys():
        if int(epoch) in epoch_filer:
            indeces = np.argsort(IS_counts[epoch])[::-1]
            outlier_perc = dict()
            for i, perc in enumerate(range(1000, len(indeces)+ 501, 500)):
                mask = np.isin(indeces[:perc], corr_indeces)
                outlier_perc[perc] = len(indeces[:perc][mask])/len(indeces[:perc])

            p_count, p_sampled = [], []
            for key, value in outlier_perc.items():
                p_count.append(key)
                p_sampled.append(value)

            plt.plot(p_count, p_sampled, label= 'Epoch ' + epoch)
    plt.legend(), plt.grid(), plt.title(title), plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title}.pdf'), dpi=2560, bbox_inches='tight')
    plt.close()

def print_hist(save_path, info, IS_counts, corr_indeces):
    os.makedirs(save_path, exist_ok=True)
    epoch_filer = [0, 61, 121, 179]
    title = info['sampler'] + '_' + 'Temp' + '_' + str(info['temperature']).replace('.', '_') + '_' + 'norm_' + str(info['normalizer'])
    fig, axs = plt.subplots(1, len(epoch_filer))
    fig.set_size_inches(18.5, 10.5)
    for epoch in IS_counts.keys():
        if int(epoch) in epoch_filer:
            i = int(np.where(epoch_filer == np.array(int(epoch)))[0])
            counts_removed_corr = np.delete(IS_counts[epoch], corr_indeces)
            counts_corr = IS_counts[epoch][corr_indeces]
            axs[i].hist(counts_removed_corr, alpha=0.5, label='Plain labels')
            axs[i].hist(counts_corr, alpha=0.5, label='Corrupted labels')
            axs[i].set_title('Epoch ' + epoch)
            # plt.legend()
            plt.grid()
            i+=1
    plt.legend(), plt.title(title), plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'distribution_{title}.pdf'))
    # plt.show()
    plt.close()

from count_visualization import get_params_from_log
# Gather counts
info_exps = []
list_IS_counts = []
i = 0
for runs in os.listdir(exp_path):
    if runs.split(f'{os.sep}')[-1] == 'analysis_experiments':
        i -= 1
        continue
    IS_counts = {}
    info_exps.append(get_params_from_log(os.path.join(exp_path, runs, 'logs', 'flogger.log')))
    for file in os.listdir(os.path.join(exp_path, runs, 'sample_counts')):
        counts_path = os.path.join(exp_path, runs, 'sample_counts', file)
        epoch = file.split('_')[-1].split('.')[0]
        f = open(counts_path, 'rb')
        IS_counts[epoch] = pkl.load(f)

    # Print histogram of counts for outliers
    print_perc(os.path.join(exp_path, runs, 'count_distr_noisy'), info_exps[-1], IS_counts, corr_indeces[i])
    print_hist(os.path.join(exp_path, runs, 'count_distr_noisy'), info_exps[-1], IS_counts, corr_indeces[i])

    list_IS_counts.append(IS_counts)
    i+= 1

# Plot print_perc of a set of experiment you can choose using: run_to_pot
epoch_filer = [0, 61, 121, 179]
run_to_plot = [i for i in range(20)]
plt.figure()
for run, IS_counts in enumerate(list_IS_counts):
    if run in run_to_plot:
        for epoch in IS_counts.keys():
            if int(epoch) in epoch_filer:
                indeces = np.argsort(IS_counts[epoch])[::-1]
                outlier_perc = dict()
                for i, perc in enumerate(range(1000, len(indeces)+ 501, 500)):
                    mask = np.isin(indeces[:perc], corr_indeces[run])
                    outlier_perc[perc] = len(indeces[:perc][mask])/len(indeces[:perc])

                p_count, p_sampled = [], []
                for key, value in outlier_perc.items():
                    p_count.append(key)
                    p_sampled.append(value)

                plt.plot(p_count, p_sampled, label=epoch + ' ' + str(run))
plt.legend(), plt.grid()
save_path = os.path.join(exp_path, 'analysis_experiments')
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, f'comparison_runs_noisy_distr.pdf'), dpi=2560, bbox_inches='tight')


# Gather forgetting indeces
