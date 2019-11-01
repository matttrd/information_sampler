import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os, json
import pdb
import matplotlib.ticker as ticker
import glob2, argparse
from os.path import expanduser
import visualization_module as viz 

if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='data viz', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment directory
    parser.add_argument('--exp_dir', default='/mnt/DATA/matteo/results/information_sampler/CVPR_sampler_resnet18_cifar10', help='Experiment directory')
    # quantity to plot on the y-axis (plot) and phase
    parser.add_argument('--plot', required=True, help='Plot to be created ("loss", "top1", "class_count")')
    parser.add_argument('--epochs', nargs='*', help='List of epochs to be considered (for class count hist)')
    opt = vars(parser.parse_args())
    
    # PLOTS
    if opt['plot'] in ['loss', 'top1']:
        viz.single_plot_train_val(opt)
    elif opt['plot'] == 'class_count':
        viz.single_class_count_hist(opt)
    else:
        NotImplementedError

