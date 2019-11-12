import argparse
import os
import random
import shutil
import time
import warnings
import sys
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from exptutils import *
from loader import data_ingredient, load_data, get_dataset_len
from sacred import Experiment
import threading
from hook import *
import pickle as pkl
from utils_lt import shot_acc
from IPython import embed
import matplotlib.pyplot as plt
import defaults
import seaborn as sns

# local thread used as a global context
ctx = threading.local()
ex = Experiment('information sampler', ingredients=[data_ingredient])
ctx.ex = ex


@ctx.ex.config
def cfg():
    '''
    Base configuration
    '''
    # batch size
    b = 128
    name_ptr = ''
    # seed
    seed = 42
    # gpu index (or starting gpu idx)
    g = int(0)
    # learning schedule
    #save model
    save = False
    # output dir
    o = f'..{os.sep}results{os.sep}'
    #whitelist for filename
    whitelist = '[]'
    # marker
    marker = ''
    exp = 'MD' # experiment ID
 
best_top1 = 0

# for some reason, the db must me created in the global scope
# that is there is not chance to select it easily (fix it)
# if ex.configurations[0]()['dbl']:
# from sacred.observers import MongoObserver
# from sacred.utils import apply_backspaces_and_linefeeds
# print('Creating database')
# ctx.ex.observers.append(MongoObserver.create())
# ctx.ex.captured_out_filter = apply_backspaces_and_linefeeds

@data_ingredient.capture
def init(name):
    ctx.global_iters = 0
    ctx.epoch = 0
    ctx.opt = init_opt(ctx)
    ctx.opt['dataset'] = name
    ctx.metrics = dict()

def get_outputs(val_loader, model):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for i, (input, target, _) in tqdm(enumerate(val_loader)):
            input = input.cuda(ctx.opt['g'])
            target = target.cuda(ctx.opt['g'])

            # compute output
            output, _ = model(input)

            #preds.append(output.max(dim=1)[1])
            preds.append(output)
            targets.append(target)

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    
    return preds, targets


def main_worker(opt):
    val_loader = None
    exp_path = os.path.join(f'..{os.sep}results', opt['exp'])
    for runs in os.listdir(exp_path):
        tmp = os.path.join(exp_path,runs)
        if runs.split(f'{os.sep}')[-1] == 'analysis_experiments' or not os.path.isdir(tmp):
            continue
        
        save_path = os.path.join(exp_path, runs, 'analysis')

        model_dict = torch.load(os.path.join(exp_path, runs, 'last_model.pth.tar'), map_location=lambda storage, loc: storage)
        opt_m = model_dict['opt']
        model = create_and_load_model(opt_m)
        model.load_state_dict(model_dict['state_dict'])
        model = model.cuda(opt['g'])

        cudnn.benchmark = True

        if val_loader is None:
            # Data loading code
            opt_m['cifar_imb_factor'] = None
            _, _, val_loader, _, train_length = load_data(opt=opt_m)
        
        preds, targets = get_outputs(val_loader, model)

        # hist of entropy
        sm = torch.softmax(preds, dim=1)
        entropies = - torch.sum(sm * th.log(sm), dim=1)
        plt.clf()
        sns.distplot(entropies.detach().cpu().numpy())
        sampler = opt_m['sampler']
        temp = opt_m['temperature']
        corr = opt_m['corr_labels']
        title = f'Entropy samp_{sampler}_temp_{temp}'
        if opt_m['corr_labels'] > 0:
            title += f'_corr_{corr}'
        
        plt.title(title)
        plt.xlabel('Entropy')
        plt.savefig(os.path.join(save_path, 'entropy.pdf'), 
                bbox_inches='tight', format='pdf')
        
        plt.clf()
        if opt_m['dataset'] == 'cifar10':
            # entropy by class
            classes = range(get_num_classes(opt_m))
            num_rows = int(np.sqrt(get_num_classes(opt_m)))
            num_cols = num_rows
            if num_cols * num_rows < get_num_classes(opt_m):
                num_rows += 1
            fig, axs = plt.subplots(num_rows, num_cols)
            for cl_id in classes:
                sm_cl = sm[targets==cl_id]
                entropies = - torch.sum(sm_cl * th.log(sm_cl), dim=1)
                plt.clf()
                sns.distplot(entropies.detach().cpu().numpy())
                title = f'Entropy cl_{cl_id} samp_{sampler}_temp_{temp}'
                if opt_m['corr_labels'] > 0:
                    title += f'_corr_{corr}'

                plt.title(title)
                plt.xlabel('Entropy')
                plt.savefig(os.path.join(save_path, f'entropy_cl_{cl_id}.pdf'), 
                    bbox_inches='tight', format='pdf')
        plt.close()
        
@ctx.ex.automain
def main():
    init()

    if ctx.opt['seed'] is not None:
        random.seed(ctx.opt['seed'])
        torch.manual_seed(ctx.opt['seed'])
        #cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')
    main_worker(ctx.opt)

    