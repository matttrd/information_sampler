import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os, json
import pdb
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import glob2, argparse 
import itertools
import numpy as np
import re
import pickle as pkl 

sns.set('paper')
sns.set_color_codes()
colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown",
          "tab:pink","tab:gray","tab:olive","tab:cyan"] 
#colors = ['b','r','g','c']

blacklist = ['lrs', 'B', 'b', 'd', 's', 'ids', 'd', \
            'save', 'metric', 'nc', 'g', 'j', 'env', 'burnin', \
            'alpha', 'delta', 'beta', 'lambdas', 'append', \
            'ratio', 'bfrac', 'l2', 'v', 'frac', 'rhos']

def set_size():
    fsz = 20
    plt.rc('text', usetex=True)
    plt.rc('font', size=fsz)
    plt.rc('axes', titlesize=fsz)
    plt.rc('axes', labelsize=fsz)
    plt.rc('xtick', labelsize=fsz)
    plt.rc('ytick', labelsize=fsz)
    plt.rc('legend', fontsize=fsz)
    plt.rc('figure', titlesize=fsz)

def get_params_from_log(f):
    r = {}
    for l in open(f,'r', encoding='latin1'):
        if '[OPT]' in l[:5]:
            r = json.loads(l[5:-1])
            fn = r['filename']
            for k in blacklist:
                r.pop(k, None)
            r['t'] = fn[fn.find('(')+1:fn.find(')')]
            return r
    assert len(r.keys) > 0, 'Could not find [OPT] marker in '+f

def loadlog(f, kws=None):
    logs, summary = [], []
    opt = get_params_from_log(f)
    for l in open(f):
        if '[LOG]' in l[:5]:
            logs.append(json.loads(l[5:-1]))
        elif '[SUMMARY]' in l[:9]:
            try:
                summary.append(json.loads(l[9:-1]))
            except ValueError:
                pdb.set_trace()
        else:
            try:
                s = json.loads(l)
            except:
                continue
            if s['i'] == 0:
                if not 'val' in s:
                    s['train'] = True
                summary.append(s)
            else:
                logs.append(s)
    dl, ds = pd.DataFrame(logs), pd.DataFrame(summary)
    dl['log'] = True
    ds['summary'] = True
    for k in opt:
        try:
            dl[k] = opt[k]
            ds[k] = opt[k]
        except Exception:
            pass
    d = pd.concat([dl, ds])
    if kws is not None:
        for k,v in kws.items():
            d[k] = v
    return d

def get_default_value(name):
    def_values = {'arch':'resnet10','dataset':'cifar10','b':128,'adam':False,'sampler':'default','normalizer':False,
                  'temperature':1.,'modatt':False,'dyncount':False,'adjust_classes':False,'ac_scaler':10,'bce':False,
                  'use_train_clean':False,'corr_labels':0.}
    return def_values[name]


def check_default_single(opt, keywords):
    kws = dict()
    if 'arch' not in keywords.keys():
        keywords['arch'] = 'resnet10'
        kws['arch'] = keywords['arch']
    if 'dataset' not in keywords.keys():
        keywords['dataset'] = 'cifar10'
        kws['dataset'] = keywords['dataset']
    if 'use_train_clean' not in keywords.keys():
        keywords['use_train_clean'] = False
        kws['use_train_clean'] = keywords['use_train_clean']
    
    return kws


def single_plots_imb_cifar(opt):
    exp_dir = opt['exp_dir']
    dataset_name = exp_dir.split('/')[-1].split('_')[-1]
    if dataset_name == 'cifar10':
        num_classes = 10
        max_count = 5000
    elif dataset_name == 'cifar100':
        num_classes = 100
        max_count = 500
    else:
        raise ValueError('Dataset must be "cifar10" or "cifar100".')
    
    ## CLASS COUNTS HIST PLOT
    expts = glob2.glob(f'{exp_dir}/*/')
    train_labels = np.load('./train_labels_' + dataset_name + '.npy')
    for f in expts:
        run_name = '/'.join(f.split('/')[:-1])
        fstr = '{' + f.split('{')[1].split('}')[0] + '}'
        keywords = json.loads(fstr)
        # default settings
        kws = check_default_single(opt, keywords) 
        # load data
        with open(os.path.join(f,'selected_indices.pkl'), 'rb') as d:
            selected_indices = pkl.load(d)
        with open(os.path.join(f,'sample_counts','sample_counts_' + str(opt['ep']) + '.pkl'), 'rb') as h:
            counts = pkl.load(h)
        class_count = np.zeros(num_classes)
        for i in range(len(counts)):
            sample_id = selected_indices[i]
            current_class = train_labels[sample_id]
            class_count[current_class] += counts[i]
        #max_count = max(class_count)
        # create plot
        plt.cla()
        fig, ax = plt.subplots()
        idx = np.asarray([i for i in range(num_classes)])
        ax.set_xticks(idx)
        ax.set_xticklabels(idx)
        ax.set_xlabel('Class')
        #ax.set_title('After {} epochs'.format(str(int(epochs[i])+1)))
        #ax.set_ylim(top=int(max_count + 0.01*max_count))
        ax.set_ylabel('Count')
        plt.bar(np.arange(0,num_classes), class_count)
        # save plot
        dir_ = os.path.join(run_name, 'analysis')
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        plt.savefig(os.path.join(dir_, 'sample_counts_hist_' + str(opt['ep']) + '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()

    ## DATASET HIST PLOT
    expts = glob2.glob(f'{exp_dir}/**/selected_indices.pkl')
    for f in expts:
        run_name = '/'.join(f.split('/')[:-1])
        fstr = '{' + f.split('{')[1].split('}')[0] + '}'
        keywords = json.loads(fstr)
        # default settings
        kws = check_default_single(opt, keywords)
        # load data
        with open(f, 'rb') as data:
            selected_indices = pkl.load(data)
        class_count = np.zeros(num_classes)
        for sample_id in selected_indices:
            current_class = train_labels[sample_id]
            class_count[current_class] += 1
        # create plot
        plt.cla()
        fig, ax = plt.subplots()
        idx = np.asarray([i for i in range(num_classes)])
        ax.set_xticks(idx)
        ax.set_xticklabels(idx)
        ax.set_xlabel('Class')
        #ax.set_title('After {} epochs'.format(str(int(epochs[i])+1)))
        ax.set_ylim(top=int(max_count + 0.01*max_count))
        ax.set_ylabel('Count')
        plt.bar(np.arange(0,num_classes), class_count)
        # save plot
        dir_ = os.path.join(run_name, 'analysis')
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        plt.savefig(os.path.join(dir_, 'dataset_hist.pdf'), bbox_inches='tight', format='pdf')
        plt.close()

    ## ACCURACY PLOT
    expts = glob2.glob(f'{exp_dir}/**/accuracies.pkl')
    for f in expts:
        run_name = '/'.join(f.split('/')[:-2])
        fstr = '{' + f.split('{')[1].split('}')[0] + '}'
        keywords = json.loads(fstr)
        # default settings
        kws = check_default_single(opt, keywords)
        # load data
        with open(f, 'rb') as data:
            acc_per_class = pkl.load(data)
        acc_per_class_last_epoch = acc_per_class[-1, :]
        # create plot
        plt.cla()
        fig, ax = plt.subplots()
        idx = np.asarray([i for i in range(num_classes)])
        ax.set_xticks(idx)
        ax.set_xticklabels(idx)
        ax.set_xlabel('Class')
        #ax.set_title('After {} epochs'.format(str(int(epochs[i])+1)))
        ax.set_ylim(top=1.05)
        ax.set_ylabel('Accuracy')
        plt.bar(np.arange(0,num_classes), acc_per_class_last_epoch)
        # save plot
        dir_ = os.path.join(run_name, 'analysis')
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        plt.savefig(os.path.join(dir_, 'acc_per_class.pdf'), bbox_inches='tight', format='pdf')
        plt.close()


if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='data viz imb cifar', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment directory
    parser.add_argument('--exp_dir', default='/mnt/DATA/Mattia_data/results/CVPR_sampler_imb_resnet18_cifar10', help='Experiment directory')
    parser.add_argument('--ep', default=179, help='Epoch for count hist')
    opt = vars(parser.parse_args())
    
    single_plots_imb_cifar(opt)


