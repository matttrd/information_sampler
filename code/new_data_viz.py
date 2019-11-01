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
    parser.add_argument('--plot', required=True, help='Plot to be created ("loss", "top1", "class_count")')
    parser.add_argument('--phase', help='Phase to plot ("train", "val", "train_val")')
    parser.add_argument('--epochs', nargs='*', help='List of epochs to be considered (for class count hist)')
    # comparison at plot level (grouping variables)
    parser.add_argument('--hue', default=None, type=json.loads, help='Hue grouping variable for the seaborn plot')
    parser.add_argument('--style', default=None, type=json.loads, help='Style grouping variable for the seaborn plot (ignored when using phase="train_val")')
    parser.add_argument('--size', default=None, type=json.loads, help='Size grouping variable for the seaborn plot')
    # comparison at file level
    parser.add_argument('--cfl', default=None, type=json.loads, help='List of variables for comparisons at file level')
    opt = vars(parser.parse_args())
    

    # when phase='train_val', we use plain line for val and dashed line for train
    if opt['phase'] == 'train_val':
        opt['style'] = None
        print('\n"style" argument ignored since "phase"=="train_val"\n')
    # when plot='class_count', opt['style'] and opt['size'] cannot be used
    if opt['plot'] == 'class_count' and opt['style'] is not None and opt['size'] is not None:
        opt['style'] = None
        opt['size'] = None
        print('\n"style" and "size" arguments cannot be used since "plot"=="class_count"\n')
    # if 'arch' and 'dataset' are not grouping variables,
    # information about the architecture and dataset must be included in 'cfl'
    options = [list(opt[name].keys())[0] for name in ['hue','style','size'] if opt[name]]
    if 'arch' not in options:
        assert opt['cfl'] is not None, "'arch' must be added to the variables for comparison at file level" 
        assert 'arch' in opt['cfl'].keys(), "'arch' must be added to the variables for comparison at file level"
    if 'dataset' not in options:
        assert opt['cfl'] is not None, "'dataset' must be added to the variables for comparison at file level"
        assert 'dataset' in opt['cfl'].keys(), "'dataset' must be added to the variables for comparison at file level"
    # PLOTS
    if opt['plot'] in ['loss', 'top1']:
        if opt['phase'] in ['train','val']:
            viz.compared_plot(opt)
        elif opt['phase'] == 'train_val':
            viz.compared_plot_train_val(opt)
    elif opt['plot'] == 'class_count':
        viz.compared_class_count_hist(opt)
    else:
        NotImplementedError

