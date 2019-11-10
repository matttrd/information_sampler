import pickle as pkl
import os
from IPython import embed
from torchvision import datasets, transforms
from loader import  load_data
from torch.utils.data import Dataset
import argparse
import numpy as np
import pandas as pd
import json

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

df = pd.DataFrame()
dfs_counts = []
dfs_fstats = []

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
    def get_sep_counts(IS_counts, corr_indeces):
        counts_removed_corr = np.delete(IS_counts, corr_indeces)
        counts_corr = IS_counts[corr_indeces]
        return counts_removed_corr, counts_corr

    os.makedirs(save_path, exist_ok=True)
    epoch_filer = [0, 61, 121, 179]
    title = info['sampler'] + '_' + 'Temp' + '_' + str(info['temperature']).replace('.', '_') + '_' + 'norm_' + str(info['normalizer'])
    # fig, axs = plt.subplots(1, len(epoch_filer))
    fig, axs = plt.subplots(1, 2 * len(epoch_filer) - 1)
    fig.set_size_inches(18.5, 10.5)
    for epoch in IS_counts.keys():
        if int(epoch) in epoch_filer:
            i = int(np.where(epoch_filer == np.array(int(epoch)))[0])
            # counts_removed_corr = np.delete(IS_counts[epoch], corr_indeces)
            # counts_corr = IS_counts[epoch][corr_indeces]
            counts_removed_corr, counts_corr = get_sep_counts(IS_counts[epoch], corr_indeces)
            axs[i].hist(counts_removed_corr, alpha=0.5, label='Plain labels')
            axs[i].hist(counts_corr, alpha=0.5, label='Corrupted labels')
            axs[i].set_title('Epoch ' + epoch)

            # first difference count (skip first)
            if int(epoch) != epoch_filer[0]:
                diff_counts = IS_counts[epoch] - IS_counts[str(epoch_filer[i-1])]
                diff_counts_removed_corr, diff_counts_corr = get_sep_counts(diff_counts, corr_indeces)
                axs[i+3].hist(diff_counts_removed_corr, alpha=0.5, label='Plain labels')
                axs[i+3].hist(diff_counts_corr, alpha=0.5, label='Corrupted labels')
                axs[i+3].set_title(f'Epoch {epoch} - {str(epoch_filer[i-1])}')
            # plt.legend()
            plt.grid()
            i+=1
    
    for ax in axs.flat:
        ax.set(xlabel='counter')
    plt.legend(), plt.tight_layout()#, plt.suptitle(title)
    plt.savefig(os.path.join(save_path, f'distribution_{title}.pdf'))
    # plt.show()
    plt.close()

def init_df(info):
    df = pd.DataFrame()
    for k in info:
        try:
            df[k] = info[k]
        except Exception:
            pass

    whitelist = ['temperature','sampler','corr_labels']
    df = df.filter(items=whitelist)
    return df

from count_visualization import get_params_from_log
info_exps = []
list_IS_counts = []
i = 0
for runs in os.listdir(exp_path):
    if runs.split(f'{os.sep}')[-1] == 'analysis_experiments':
        i -= 1
        continue
    IS_counts = {}
    info = get_params_from_log(os.path.join(exp_path, runs, 'logs', 'flogger.log'))
    info_exps.append(info)
    corr_perc = info['corr_labels']
    best = 0.

    for file in os.listdir(os.path.join(exp_path, runs, 'sample_counts')):
        counts_path = os.path.join(exp_path, runs, 'sample_counts', file)
        epoch = file.split('_')[-1].split('.')[0]
        f = open(counts_path, 'rb')
        IS_counts[epoch] = pkl.load(f)
        corr_ind_set = set(corr_indeces[i])
        len_ = len(corr_ind_set)
        sorted_indices = np.argsort(IS_counts[epoch])
        predicted_outliers = sorted_indices[-len_:]
        assert len(set(predicted_outliers)) == len(corr_ind_set)
        perc_spotted_outliers = len(set(predicted_outliers) - corr_ind_set) / len(corr_ind_set)
    
        if perc_spotted_outliers > best:
            best = perc_spotted_outliers

    df = init_df(info)
    # df['perc_outlier'] = corr_perc
    df['count_best'] = best
    df['count_last'] = perc_spotted_outliers
    dfs_counts.append(df.copy())
    # Print histogram (also differential) of counts for outliers
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
from analyze_forgetting_stats import compute_forgetting_statistics, sort_examples_by_forgetting
def forg_stats_based_spotted_outliers(run_path, corr_indeces, corr_perc):
    try:
        f = open(os.path.join(run_path, 'forgetting_stats', 'stats.pkl'), 'rb')
        loaded = pkl.load(f)
        # Compute the forgetting statistics per example for training run
        _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(
            loaded, 200) # note 200 is the max number of epochs

        ordered_examples, ordered_values = sort_examples_by_forgetting(
                unlearned_per_presentation, first_learned, 200)
        corr_ind_set = set(corr_indeces)
        len_ = len(corr_ind_set)
        predicted_outliers = ordered_examples[-len_:]
        assert len(set(predicted_outliers)) == len(corr_ind_set)
        perc_spotted_outliers = len(set(predicted_outliers) - corr_ind_set) / len(corr_ind_set)
        return perc_spotted_outliers
    except FileNotFoundError:
        return None

i = 0
for runs in os.listdir(exp_path):
    path = os.path.join(exp_path, runs)
    if runs.split(f'{os.sep}')[-1] == 'analysis_experiments':
        i -= 1
        continue
    info_exp = get_params_from_log(os.path.join(path, 'logs', 'flogger.log'))

    corr_ind = corr_indeces[i]
    pso = forg_stats_based_spotted_outliers(path, corr_ind, info_exp['corr_labels'])
    df = init_df(info_exp)
    # df['perc_outlier'] = corr_perc
    df['fstats'] = pso
    dfs_fstats.append(df.copy())
    i+=1

table_counts = pd.concat(dfs_counts)
table_fstats = pd.concat(dfs_fstats)

table_counts = pd.pivot_table(table_counts, index=['sampler','temperature'], columns='corr_labels', aggfunc='first')
table_fstats = pd.pivot_table(table_fstats, index=['sampler','temperature'], columns='corr_labels', aggfunc='first')

#export to pickle
table_counts.to_pickle(os.path.join(save_path, 'corr_table_counts.pickle'))
table_fstats.to_pickle(os.path.join(save_path, 'corr_table_fstats.pickle'))

# export to latex
table_counts.to_latex(os.path.join(save_path, 'corr_table_counts.tex'), multirow=True)
table_fstats.to_latex(os.path.join(save_path, 'corr_table_fstats.tex'), multirow=True)

