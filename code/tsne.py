import argparse
import os
import random
import shutil
import time
import warnings
import sys
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchnet.meter import ClassErrorMeter, ConfusionMeter, TimeMeter, AverageValueMeter
from exptutils import *
#import torchvision.models as models
from loader import load_data, get_dataset_len
from sacred import Experiment
import threading
from hook import *
import pickle as pkl
from utils_lt import shot_acc
from IPython import embed
from PIL import Image
from sklearn.decomposition import PCA
from sklearn import manifold
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='tsne', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# experiment directory
parser.add_argument('--checkpoint', help='model path')
parser.add_argument('--dataset', default='cifar10', help='dataset')
parser.add_argument('--logits',  help='use logits for output',  action='store_true')
parser.add_argument('--pca',  help='use pca',  action='store_true')
parser.add_argument('--npca', default=100, help='number of principal components')
parser.add_argument('-g', default=0, help='gpu index')
parser.add_argument('-b', default=512, help='batch_size')


# quantity to plot on the y-axis (plot) and phase
opt = vars(parser.parse_args())


def get_feat(val_loader, model, opt):
    model.eval()
    targets = []
    hidden_reps = []
    sig = nn.Softmax()
    sig = sig.cuda(opt['g'])
    with torch.no_grad():
        for i, (input, target, _) in tqdm(enumerate(val_loader)):
            input = input.cuda(opt['g'])
            target = target.cuda(opt['g'])
            # compute output
            output, feat = model(input)
            if opt['logits']:
                logits = sig(output)
                hidden_reps.append(logits)
            else:
                hidden_reps.append(feat)
            targets.append(target)
        hidden_reps = torch.cat(hidden_reps)
        targets = torch.cat(targets)
    return hidden_reps.cpu().numpy(), targets.cpu().numpy()


def plot_tsne(x, y, npca=50, use_pca=True, markersize=10, opts=None):
    if use_pca:
        X = PCA(n_components=npca).fit_transform(x)
    else:
        X = x
    Y = manifold.TSNE(n_components=2, init='pca').fit_transform(X)
    palette = sns.color_palette("Paired", len(np.unique(y)))
    color_dict = {l: c for l, c in zip(range(len(np.unique(y))), palette)}
    colors = [color_dict[l] for l in y]
    fig, ax = plt.subplots()
    scatter = ax.scatter(Y[:, 0], Y[:, 1], markersize, c=colors, marker='o')
    #legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    #ax.add_artist(legend1)
    #plt.show()
    fn = '_'.join(('/home/matteo/Dropbox/results/tsne', opts['sampler'], opts['dataset'], opts['arch']))
    if opts['sampler'] is not 'default':
        fn = fn + '_' + str(opts['temperature']) + '.pdf'
    plt.savefig(fn, format='pdf')

def main():
    checkpoint = torch.load(opt['checkpoint'])
    opt_m = checkpoint['opt']
    opt_m['dataset'] = opt['dataset']
    opt_m['b'] = opt['b']
    model = create_and_load_model(opt_m)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda(opt['g'])
    _, _, test_loader, _, _ = load_data(name=opt['dataset'], 
                                        source='../data/', 
                                        shuffle=False, 
                                        frac=1, 
                                        perc=0, 
                                        mode=0,
                                        pilot_samp='default', 
                                        pilot_arch='resnet10', 
                                        mode_source=0, 
                                        num_clusters=500, 
                                        use_perc_diff=False, 
                                        celeba_class_attr='Smiling', 
                                        norm=False, 
                                        opt=opt_m)
    
    hidden_def, targets = get_feat(test_loader, model, opt)
    plot_tsne(hidden_def, targets, npca=opt['npca'], use_pca=opt['pca'], opts=opt_m)


if __name__ == '__main__' :
    main()
