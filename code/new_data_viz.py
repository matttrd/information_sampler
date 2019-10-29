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
    # reference directories
    parser.add_argument('--sd', default='/home/mattiacarletti/Desktop/plots/', help='Save directory')
    parser.add_argument('--base', default='/mnt/DATA/Mattia_data/results/', help='Base directory')
    # the following 3 parameters define the sub-directories from which we load data
    parser.add_argument('--exp', required=True, help='Experiment to be considered')
    parser.add_argument('--arch', required=True, nargs='*', help='List of architectures to be considered ("resnet18", "resnet34")')
    parser.add_argument('--datasets', required=True, nargs='*', help='List of datasets to be considered ("cifar10", "cifar100", "imagenet_lt")')
    # quantity to plot on the y-axis (plot) and phase
    parser.add_argument('--plot', required=True, help='Plot to be created ("loss", "top1")')
    parser.add_argument('--phase', required=True, help='Phase to plot ("train", "val", "train_val")')
    # comparison at plot level (grouping variables)
    parser.add_argument('--hue', default=None, help='Hue grouping variable for the seaborn plot')
    parser.add_argument('--style', default=None, help='Style grouping variable for the seaborn plot (ignored when using phase="train_val")')
    parser.add_argument('--size', default=None, help='Size grouping variable for the seaborn plot')
    # comparison at file level
    parser.add_argument('--cfl', default=None, nargs='*', help='List of variables for comparisons at file level')
    opt = vars(parser.parse_args())
    

    # when phase='train_val', we use plain line for val and dashed line for train
    if opt['phase'] == 'train_val':
        opt['style'] = None
        print('\n"style" argument ignored since "phase"=="train_val"\n')
    # if 'arch' and 'dataset' are not grouping variables,
    # information about the architecture and dataset must be included in 'cfl'
    if 'arch' not in [opt['hue'],opt['style'],opt['size']]:
        assert opt['cfl'] is not None, "'arch' must be added to the variables for comparison at file level" 
        assert 'arch' in opt['cfl'], "'arch' must be added to the variables for comparison at file level"
    if 'dataset' not in [opt['hue'],opt['style'],opt['size']]:
        assert opt['cfl'] is not None, "'dataset' must be added to the variables for comparison at file level"
        assert 'dataset' in opt['cfl'], "'dataset' must be added to the variables for comparison at file level"
    # PLOTS
    if opt['plot'] in ['loss', 'top1']:
        if opt['phase'] in ['train','val']:
            viz.plot(opt)
        elif opt['phase'] == 'train_val':
            viz.plot_train_val(opt)
    else:
        NotImplementedError

