'''
python computing_gradients_stats.py with dataset.name='cifar10' arch='resnet18' exp='CVPR_test_sampler' run='3' resume='model_159.pth.tar' b=512
'''

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
from loader import data_ingredient, load_data, get_dataset_len
from sacred import Experiment
import threading
from hook import *
import pickle as pkl
from utils_lt import shot_acc
from IPython import embed
import matplotlib.pyplot as plt
import defaults
from torchvision import datasets, transforms
from loader import MyDataset

# local thread used as a global context
ctx = threading.local()
ex = Experiment('information sampler', ingredients=[data_ingredient])
ctx.ex = ex

from main_CVPR import  cfg, cum_abs_diff, runner, train, main_worker

@ctx.ex.config
def cfg():
    '''
    Base configuration
    '''
    # architecture
    arch = 'resnet10'
    # number of data loading workers (default: 4)
    j = 1
    # number of total B to run
    epochs = 200
    # starting epoch
    start_epoch = 0
    # batch size
    b = 128
    # initial learning rate
    lr = 0.1
    # momentum
    momentum = 0.9
    #nesterov
    nesterov = True
    # weight decay (ell-2)
    wd = 0.
    # print-freq
    print_freq = 50
    # path to latest checkpoint
    resume = ''
    # evaluate a model on validation set
    evaluate = False
    # if True, load a pretrained model
    pretrained = False
     # if pretrained= True, load 'name_ptr' model
    name_ptr = ''
    # seed
    seed = 42
    # gpu index (or starting gpu idx)
    g = int(0)
    # number of gpus for data parallel
    ng = int(0)
    # learning schedule
    lrs = '[[0,0.1],[60,0.02],[120,0.004],[160,0.0008],[180,0.0001]]'
    adam = False
    # dropout (if any)
    d = 0.
    #save model
    save = False
    save_epochs = []
    # file logger
    if save:
        fl = True
    else:
        fl = False
    # Tensorflow logger
    tfl = False
    dbl = True
    # output dir
    o = f'..{os.sep}results{os.sep}'
    #whitelist for filename
    whitelist = '[]'
    # marker
    marker = ''
    unbalanced = False
    sampler = 'default' # (default | invtunnel | tunnel | class )
    normalizer = False
    # tunneling temperature
    temperature = 1.
    temperatures = ''
    wufreq = 1 #weights sampler frequency
    classes = None
    modatt = False # modulated attention
    dyncount = False
    adjust_classes = False
    ac_scaler = 10 # adjust classes scaler hyperpar
    pilot = False # if True, pilot net mode (min dataset exp) and save indices according to their weight
    exp = 'MD' # experiment ID
    bce = False #binary xce
    use_train_clean = False # use the clean_train_loader to validate on the training set
    freq_save_counts = 20 # frequency of sample counts save (no save if equal to 0)
    save_counts_list = [0,59,60,61,119,120,121,159,160,161]
    corr_labels = 0. # fraction of labels to be corrupted
    run = None

@data_ingredient.capture
def init(name):
    ctx.global_iters = 0
    ctx.epoch = 0
    ctx.opt = init_opt(ctx)
    ctx.opt = defaults.add_args_to_opt(name, ctx.opt)

    if ctx.opt.get('filename', None) is None:
        build_filename(ctx)

    ctx.opt['dataset'] = name
    ctx.opt['exp'] = '_'.join([ctx.opt['exp'], ctx.opt['arch'], ctx.opt['dataset']])
    ctx.metrics = dict()
    ctx.metrics['best_top1'] = best_top1
    ctx.hooks = None
    ctx.init = 0
    ctx.counter = 0
    ctx.forgetting_stats = None #Used to store all the interesting statistics for the forgetting experiment
    # if ctx.opt['sampler'] == 'our':
    #     ctx.weights_logger = create_basic_logger(ctx, 'statistics', idx=0)
    register_hooks(ctx)

best_top1=0

def get_save_grad_stats(train_loader, S_loading, model, optimizer, loaded_model_name):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda(ctx.opt['g'])
    model.train()
    gradients = []
    for i, (input, target, idx) in enumerate(train_loader):
        input, target = input.cuda(ctx.opt['g']), target.cuda(ctx.opt['g'])

        output, _ = model(input)
        loss = criterion(output, target).mean()

        optimizer.zero_grad()
        loss.backward()

        grad = []
        for i, param in enumerate(model.parameters()):
            grad.append(param.grad.data.cpu().numpy().reshape(-1,))

        gradients.append(np.concatenate(grad))

    # problay this line can be replaced by something more efficient
    gradients = np.array(gradients)

    mean_grad = gradients.mean(axis= 0)
    var_grad = gradients.var(axis= 0)
    cosine_director = []
    gradients_norm = []
    normalized_mean_grad = mean_grad/np.linalg.norm(mean_grad)
    for i in range(gradients.shape[0]):
        gradients_norm.append(np.linalg.norm(gradients[i,:]))
        cosine_director.append(np.dot(gradients[i,:]/gradients_norm[i], normalized_mean_grad))
        # cosine_director.append(np.dot(gradients[i,:], mean_grad)

    cosine_director = np.array(cosine_director)
    gradients_norm = np.array(gradients_norm)

    data_state = {
                      'model_dimension':  mean_grad.size,
                      'sum_mean_grad':    np.sum(mean_grad),
                      'sum_var_grad':     np.sum(var_grad),
                      'cosine_directors': cosine_director,
                      'gradients_norm':   gradients_norm
                  }

    # Saving statistics
    saving_path = os.path.join('..', 'results', ctx.opt['exp'], ctx.opt['run'],
                               'gradients_stats')
    os.makedirs(saving_path, exist_ok=True)
    file_path = os.path.join(saving_path, f"{S_loading}_BS_{ctx.opt['b']}_" + loaded_model_name.split('.')[0] + '.pkl')
    with open(file_path, 'wb') as file:
        pkl.dump(data_state, file, protocol=pkl.HIGHEST_PROTOCOL)



@ctx.ex.automain
def main():
    global best_top1
    init()
    # ctx.opt = init_opt(ctx)

    if ctx.opt['seed'] is not None:
        random.seed(ctx.opt['seed'])
        torch.manual_seed(ctx.opt['seed'])
        #cudnn.deterministic = True
        model = create_and_load_model(ctx.opt)

        torch.cuda.set_device(ctx.opt['g'])
        model = model.cuda(ctx.opt['g'])

        optimizer = torch.optim.SGD(model.parameters(), lr=0)

        ctx.model = model
        loaded_model_name = ctx.opt['resume']
        ctx.opt['resume'] = os.path.join('..', 'results', ctx.opt['exp'], ctx.opt['run'], ctx.opt['resume'])
        # optionally resume from a checkpoint
        if ctx.opt['resume']:
            if os.path.isfile(ctx.opt['resume']):
                print("=> loading checkpoint '{}'".format(ctx.opt['resume']))
                checkpoint = torch.load(ctx.opt['resume'], map_location='cuda:'+ str(ctx.opt['g']))
                ctx.opt['start_epoch'] = checkpoint['epoch']
                best_top1 = checkpoint['best_top1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(ctx.opt['resume'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(ctx.opt['resume']))

        cudnn.benchmark = True

        if ctx.opt['dataset'] == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif ctx.opt['dataset'] == 'cifar100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])

            transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        train_dataset = MyDataset(ctx.opt['dataset'], f'..{os.sep}data{os.sep}', train=True, download=True,
                                                      transform=transform_train)
        test_dataset = MyDataset(ctx.opt['dataset'], f'..{os.sep}data{os.sep}', train=False, download=True,
                                            transform=transform_test)


        try:
            # Creating loaders train sampler, train plain, val
            count_path = os.path.join('..', 'results', ctx.opt['exp'], ctx.opt['run'],
                                            'weigths_folder', 'S_weights_' + loaded_model_name.split('.')[0].split('model_')[1] + '.pkl')
            with open(count_path, 'rb') as file:
                S_weights = pkl.load(file)


            S_weights = torch.DoubleTensor(S_weights)
            sampler = torch.utils.data.WeightedRandomSampler(weights=S_weights,
                                                             num_samples=int(len(S_weights)),
                                                             replacement=True)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=ctx.opt['b'], shuffle=False,
                                            num_workers=ctx.opt['j'], pin_memory=False,
                                            sampler=sampler)
            get_save_grad_stats(train_loader, 'impsamp', model, optimizer, loaded_model_name)
        except:
            print(f"Experiment {ctx.opt['exp']} does not possess saved sampling weights!!!")

        Unif_weights = torch.DoubleTensor(np.zeros(len(train_dataset)) + 1)
        unif_sampler = torch.utils.data.WeightedRandomSampler(weights=Unif_weights,
                                                              num_samples=int(len(Unif_weights)),
                                                              replacement=False)
        train_loader_unif = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=ctx.opt['b'], shuffle=False,
                                            num_workers=ctx.opt['j'],pin_memory=False,
                                            sampler=unif_sampler)



        get_save_grad_stats(train_loader_unif, 'default', model, optimizer, loaded_model_name)
