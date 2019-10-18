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
import models
from loader import data_ingredient, load_data, get_dataset_len
from sacred import Experiment
import threading
from hook import *
import pickle as pkl
import cifar_models
import imagenet_models
import celeba_models
from utils_lt import shot_acc
from IPython import embed
import matplotlib.pyplot as plt 


# local thread used as a global context
ctx = threading.local()
ex = Experiment('information sampler', ingredients=[data_ingredient])
ctx.ex = ex

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


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
    # dropout (if any)
    d = 0.
    #save model
    save = False

    # file logger
    if save:
        fl = True
    else:
        fl = False
    # Tensorflow logger
    tfl = False
    dbl = True
    # output dir
    o = '../results/'
    #whitelist for filename
    whitelist = '[]'
    # marker
    marker = ''
    unbalanced = False
    sampler = 'default' # default | tunnel | invtunnel | alternate
    # tunneling temperature
    temperature = 1.
    temperatures = ''
    wufreq = 1 #weights sampler frequency
    topkw = 500 # number of weight to analyse (default 500)
    classes = None
    modatt = False # modulated attention
    dyncount = False
    adjust_classes = False
    pilot = False # if True, pilot net mode (min dataset exp) and save indices according to their weight
    exp = 'MD' # experiment ID
    save_w_dyn = False
    bce = False #binary xce
    smart_init_sampler = False 
    clustering = False
    mg_iter = 1 # multi-grad steps (if equal to 1 -> standard training)
    save_hist_until_ep = 0 # if > 0, save histograms until the specified epoch
    ep_drop = -1 # if != -1, epoch in which hard samples are filtered out
    num_drop = 50 # number of hard samples to filter out
    use_train_clean = False # use the clean_train_loader to validate on the training set
    num_tunnel = 1 # if sampler=alternate, number of epochs with "tunnel" sampler
    num_invtunnel = 1 # if sampler=alternate, number of epochs with "invtunnel" sampler
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
    if ctx.opt.get('filename', None) is None:
        build_filename(ctx)

    ctx.opt['dataset'] = name
    ctx.opt['exp'] = '_'.join([ctx.opt['exp'], ctx.opt['arch'], ctx.opt['dataset']])
    ctx.metrics = dict()
    ctx.metrics['best_top1'] = best_top1
    ctx.hooks = None
    ctx.toweights = {'indices': [], 'values': []}
    ctx.init = 0
    ctx.counter = 0
    ctx.counter_tunnel = ctx.opt['num_tunnel']
    ctx.counter_invtunnel = 0
    register_hooks(ctx)
    
def cum_abs_diff(cumsum, x_prev, x_new):
    return cumsum + torch.abs(x_prev - x_new)

def compute_weights(criterion, weights_loader, model, opt):
    sampler = opt['sampler']
    weights = []
    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(weights_loader):
            data, target = data.cuda(opt['g']), target.cuda(opt['g'])
            output, _ = model(data)
            if ctx.opt['bce']:
                new_target = logical_index(target, output.shape).float()
            else:
                new_target = target
            loss = criterion(output, target)
            if sampler == 'invtunnel':
                w = 1 - torch.exp(-loss / ctx.opt['temperature'])
            elif sampler == 'tunnel':
                w = torch.exp(-loss / ctx.opt['temperature'])
            else:
                raise NotImplementedError    
            weights.append(w)
    S_prob = torch.cat(weights)

    return S_prob


def compute_weights_alt(criterion, weights_loader, model, opt):
    if ctx.counter_invtunnel != 0 and ctx.counter_tunnel == 0:
        sampler = 'invtunnel'
    elif ctx.counter_tunnel != 0 and ctx.counter_invtunnel == 0:
        sampler = 'tunnel'

    weights = []
    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(weights_loader):
            data, target = data.cuda(opt['g']), target.cuda(opt['g'])
            output, _ = model(data)
            if ctx.opt['bce']:
                new_target = logical_index(target, output.shape).float()
            else:
                new_target = target
            loss = criterion(output, target)
            if sampler == 'invtunnel':
                w = 1 - torch.exp(-loss / ctx.opt['temperature'])
            elif sampler == 'tunnel':
                w = torch.exp(-loss / ctx.opt['temperature'])    
            weights.append(w)
    S_prob = torch.cat(weights)

    if sampler == 'invtunnel':
        ctx.counter_invtunnel = ctx.counter_invtunnel - 1
        if ctx.counter_invtunnel == 0:
            ctx.counter_tunnel = ctx.opt['num_tunnel']
    if sampler == 'tunnel':
        ctx.counter_tunnel = ctx.counter_tunnel - 1
        if ctx.counter_tunnel == 0:
            ctx.counter_invtunnel = ctx.opt['num_invtunnel']

    return S_prob

def compute_weights_stats(model, criterion, loader, save_stats):
    opt = ctx.opt
    model.eval()
    weights = []
    s_weights = []
    target_list = []
    hist_list = []
    
    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(loader):
            data, target = data.cuda(opt['g']), target.cuda(opt['g'])
            output, _ = model(data)

            if ctx.opt['bce']:
                new_target = logical_index(target, output.shape).float()
            else:
                new_target = target

            loss = criterion(output, target)
            if ctx.opt['sampler'] == 'tunnel':
                w = torch.exp(-loss / ctx.opt['temperature'] )
            else:
                w = 1 - torch.exp(-loss / ctx.opt['temperature'] )

            weights.append(w)
            target_list.append(target)

    targets_tensor = torch.cat(target_list)
    weights = torch.cat(weights)
    sorted_, indices = torch.sort(weights)
    topk = ctx.opt['topkw']
    topk_idx = indices[:topk]
    topk_value = sorted_[:topk]
    num_classes = output.shape[1]
    ctx.toweights['indices'].append(topk_idx.data.cpu().numpy())
    ctx.toweights['values'].append(topk_value.data.cpu().numpy())
    
    if ctx.init == 0:
        ctx.histograms = {str(k): [] for k in range(num_classes)}
        ctx.histograms['total'] = []
        ctx.init = 1
    for cl in range(num_classes):
        idx_w_cl = targets_tensor == cl
        w_cl = weights[idx_w_cl]
        hist, bin_edges = np.histogram(w_cl.cpu().numpy(), bins=100, range=(0,1))
        ctx.histograms[str(cl)].append((hist,bin_edges))
    
    hist, bin_edges = np.histogram(weights.cpu().numpy(), bins=100, range=(0,1))
    ctx.histograms['total'].append((hist,bin_edges))
    # update sample mean of the weights and differences (new_weights - old_weights)
    
    if ctx.counter == 0:
        if save_stats:
            if opt['pilot']:
                inp_w_dir = os.path.join(opt.get('o'), 'pilots', opt['filename']) +'/input_weights/'
            else:
                inp_w_dir = os.path.join(opt.get('o'), opt['exp'], opt['filename']) +'/input_weights/'
            ctx.inp_w_dir = inp_w_dir
            os.makedirs(inp_w_dir)
            #os.makedirs(os.path.join(inp_w_dir, 'tmp'))
        # initialize sample mean of the weights
        ctx.sample_mean = torch.zeros([1, len(loader.dataset)]).cuda(opt['g'])     
        ctx.sample_mean = torch.zeros_like(ctx.sample_mean)
        # here will be stored weights of the last update
        ctx.old_weights = torch.zeros([1, len(loader.dataset)]).cuda(opt['g'])
        ctx.cum_sum_diff = torch.zeros_like(ctx.old_weights).cuda(opt['g'])   
        ctx.cum_sum = 0

    ctx.counter += 1
    ctx.weights = weights
    ctx.sample_mean = ((ctx.counter - 1) / ctx.counter) * ctx.sample_mean + (weights / ctx.counter)
    difference = weights - ctx.old_weights
    ctx.cum_sum_diff = cum_abs_diff(ctx.cum_sum, ctx.old_weights, weights)
    ctx.cum_sum += torch.abs(weights)

    if save_stats:
        with open(os.path.join(ctx.inp_w_dir, 'weights_' + str(ctx.counter) + '.pkl'), 'wb') as handle:
            pkl.dump(weights.cpu().numpy(), handle, protocol=pkl.HIGHEST_PROTOCOL)
            
    ctx.old_weights = weights

    return weights



@batch_hook(ctx, mode='train')
def runner(input, target, model, criterion, optimizer, idx, weights_iter, weights_loader):
    # compute output

        output, _ = model(input)

        if ctx.opt['bce']:
            orig_target = target.clone()
            new_target = logical_index(target, output.shape).float()
        else:
            new_target = target

        loss = criterion(output, new_target).mean()

        # measure accuracy and record loss
        ctx.errors.add(output.data, target.data)
        ctx.losses.add(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_stats = {'loss': ctx.losses.value()[0], 
                     'top1': ctx.errors.value()[0]}
        # batch_stats = {'loss': loss.item(), 
        #          'top1': ctx.errors.value()[0], 
        #          'top5': ctx.errors.value()[1]}

        # ctx.metrics.batch = stats
        ctx.metrics['avg'] = avg_stats
        
        model.eval()
        #S_prob = compute_weights(output, target, idx, criterion, weights_iter, weights_loader, model, ctx.opt)
        model.train()
        avg_stats['top1'] = (100 - avg_stats['top1']) / 100.
        return avg_stats

@epoch_hook(ctx, mode='train')
def train(train_loader, model, criterion, optimizer, epoch, opt, weights_iter, weights_loader):
    data_time = TimeMeter(unit=1)
    ctx.losses = AverageValueMeter()
    if opt['dataset'] == 'celeba':
        ctx.errors = ClassErrorMeter(topk=[1])
    else:
        ctx.errors = ClassErrorMeter(topk=[1,5])
    n_iters = int(len(train_loader) * opt['wufreq'])

    # switch to train mode
    model.train()

    # end = time.time()
    for i, (input, target, idx) in enumerate(train_loader):
        # tmp var (for convenience)

        ctx.i = i
        ctx.global_iters += 1
        input = input.cuda(opt['g'])
        target = target.cuda(opt['g'])
        for mg_idx in range(opt['mg_iter']):
            stats = runner(input, target, model, criterion, optimizer, idx, weights_iter, weights_loader)

        loss = stats['loss']
        top1 = stats['top1']

        if i % opt['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {time:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Acc@1 {top1:.3f}\t'.format(
                   epoch, i, len(train_loader),
                   time=data_time.value(), loss=loss, 
                   top1=top1 * 100.))

    return stats
 
@epoch_hook(ctx, mode='val')
def validate(val_loader, train_dataset, model, criterion, opt):
    losses = AverageValueMeter()
    if ctx.opt['dataset'] == 'celeba':
        errors = ClassErrorMeter(topk=[1])
    else:
        errors = ClassErrorMeter(topk=[1,5])
    # switch to evaluate mode
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for i, (input, target, _) in tqdm(enumerate(val_loader)):
            input = input.cuda(opt['g'])
            target = target.cuda(opt['g'])

            # compute output
            output, _ = model(input)

            if ctx.opt['bce']:
                orig_target = target.clone()
                new_target = logical_index(target, output.shape).float()
            else:
                new_target = target

            loss = criterion(output, new_target).mean()
            preds.append(output.max(dim=1)[1])
            targets.append(target)

            errors.add(output, target)
            losses.add(loss.item())
         
            loss = losses.value()[0]
            top1 = (100 - errors.value()[0]) / 100.

        print(' * Acc@1 {top1:.3f}'
              .format(top1=top1 * 100.))
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    
    stats = {'loss': loss, 'top1': top1}

    if ctx.opt['dataset'] == 'imagenet_lt':
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds, targets, train_dataset)
        stats['many_acc_top1'] = many_acc_top1
        stats['median_acc_top1'] = median_acc_top1
        stats['low_acc_top1'] = low_acc_top1

    if ctx.opt['dataset'] == 'celeba':
        preds_a = preds[ctx.indices_a].cpu()
        preds_b = preds[ctx.indices_b].cpu()
        targets_a = targets[ctx.indices_a].cpu()
        targets_b = targets[ctx.indices_b].cpu()

        acc_a = accuracy_score(targets_a, preds_a)
        cm_a = confusion_matrix(targets_a, preds_a)
        cm_a = cm_a.astype('float') / cm_a.sum(axis=1)[:, np.newaxis]
        tn_a, fp_a, fn_a, tp_a = cm_a.ravel()
        f1_a = f1_score(targets_a, preds_a)

        acc_b = accuracy_score(targets_b, preds_b)
        cm_b = confusion_matrix(targets_b, preds_b)
        cm_b = cm_b.astype('float') / cm_b.sum(axis=1)[:, np.newaxis]
        tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
        f1_b = f1_score(targets_b, preds_b)

        stats['acc_a'] = acc_a
        stats['fp_a'] = fp_a
        stats['fn_a'] = fn_a
        stats['f1_a'] = f1_a
        stats['acc_b'] = acc_b
        stats['fp_b'] = fp_b
        stats['fn_b'] = fn_b
        stats['f1_b'] = f1_b

        print('a) Acc@1 {acc:.3f}\t'
              'FP rate {fp:.3f}\t'
              'FN rate {fn:.3f}\t'
              'F1 score {f1:.3f}\t'.format(
              acc=acc_a * 100., fp=fp_a, fn=fn_a, f1=f1_a))
        print('b) Acc@1 {acc:.3f}\t' 
              'FP rate {fp:.3f}\t'
              'FN rate {fn:.3f}\t'
              'F1 score {f1:.3f}\t'.format(
              acc=acc_b*100., fp=fp_b, fn=fn_b, f1=f1_b))
    
    ctx.metrics = stats
    return stats

@epoch_hook(ctx, mode='train_clean')
def train_clean(val_loader, train_dataset, model, criterion, opt):
    losses = AverageValueMeter()
    if ctx.opt['dataset'] == 'celeba':
        errors = ClassErrorMeter(topk=[1])
    else:
        errors = ClassErrorMeter(topk=[1,5])
    # switch to evaluate mode
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for i, (input, target, _) in tqdm(enumerate(val_loader)):
            input = input.cuda(opt['g'])
            target = target.cuda(opt['g'])

            # compute output
            output, _ = model(input)

            if ctx.opt['bce']:
                orig_target = target.clone()
                new_target = logical_index(target, output.shape).float()
            else:
                new_target = target

            loss = criterion(output, new_target).mean()
            preds.append(output.max(dim=1)[1])
            targets.append(target)

            errors.add(output, target)
            losses.add(loss.item())
         
            loss = losses.value()[0]
            top1 = (100 - errors.value()[0]) / 100.

        print(' * Acc@1 {top1:.3f}'
              .format(top1=top1 * 100.))
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    
    stats = {'loss': loss, 'top1': top1}

    if ctx.opt['dataset'] == 'imagenet_lt':
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds, targets, train_dataset)
        stats['many_acc_top1'] = many_acc_top1
        stats['median_acc_top1'] = median_acc_top1
        stats['low_acc_top1'] = low_acc_top1

    ctx.metrics = stats
    return stats


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not ctx.opt['save']:
        return
    opt = ctx.opt
    if opt['pilot']:
        fn = os.path.join(opt['o'], 'pilots', opt['filename'], 'last_model.pth.tar')
    else:
        fn = os.path.join(opt['o'], opt['exp'], opt['filename'], 'last_model.pth.tar')
    r = gitrev(opt)
    meta = dict(SHA=r[0], STATUS=r[1], DIFF=r[2])
    state.update({'meta': meta})
    torch.save(state, fn)
    # if is_best:
    #     # filename = os.path.join(opt['o'], opt['arch'], 
    #     #                     opt['filename']) + '_best.pth.tar'
    #     filename = os.path.join(opt['o'], opt['exp'], opt['filename'], 
    #                         'best.pth.tar')
    #     shutil.copyfile(fn, filename)


# adjust learning rate and log 
def adjust_learning_rate(epoch):
    opt = ctx.opt
    optimizer = ctx.optimizer

    if opt['lrs'] == '':
        # default lr schedule
        lr = opt['lr'] * (0.1 ** (epoch // 30))
    else:
        lr = schedule(ctx, k='lr')
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('[%s]: '%'lr', lr)
    if opt.get('fl', None) and ctx.ex.logger:
        ctx.ex.logger.info('[%s] '%'lr' + json.dumps({'%s'%'lr': lr}))

def adjust_temperature(epoch, opt):
    if len(opt['temperatures']) > 0:
        temps = opt['temperatures']
        initial_temp = temps[0]
        final_temp = temps[1]
        ratio = (epoch / opt['epochs']) ** 0.5
        ctx.opt['temperature'] = final_temp * ratio + initial_temp * (1 - ratio)
    return 


@train_hook(ctx)
def main_worker(opt):
    global best_top1

     # create model
    if opt['pretrained']:
        print("=> using pre-trained model '{}'".format(opt['arch']))
        if 'allcnn' in opt['arch'] or 'wrn' in opt['arch'] or 'lenet' in opt['arch']:
            model = models.__dict__[opt['arch']](opt)
            #load_pretrained(model, opt['dataset'])
            models.load_pretrained(model, opt['pretrained'])
        else:
            model = models.__dict__[opt['arch']](opt, pretrained=True)
    else:
        if opt['dataset'] == 'cifar10' and opt['arch'] in ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            print("=> creating model '{}'".format(opt['arch']))
            model = getattr(cifar_models, opt['arch'])(num_classes=10)
        elif opt['dataset'] == 'cifar100' and opt['arch'] in ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            print("=> creating model '{}'".format(opt['arch']))
            model = getattr(cifar_models, opt['arch'])(num_classes=100)
        elif 'imagenet' in opt['dataset'] and not '32' in opt['dataset'] and opt['arch'] in ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            print("=> creating model '{}'".format(opt['arch']))
            model = getattr(imagenet_models, opt['arch'])(num_classes=1000, use_att=opt['modatt'])
        elif opt['dataset'] == 'celeba' and opt['arch'] in ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            print("=> creating model '{}'".format(opt['arch']))
            model = getattr(celeba_models, opt['arch'])(num_classes=2)
        else:
            print("=> creating model '{}'".format(opt['arch']))
            model = models.__dict__[opt['arch']](opt)

    if opt['ng'] == 0:
        torch.cuda.set_device(opt['g'])
        model = model.cuda(opt['g'])
    else:
        model = torch.nn.DataParallel(model, 
                        device_ids=range(opt['g'], opt['g'] + opt['ng']),
                        output_device=opt['g']).cuda()

    # define loss function (criterion) and optimizer
    if opt['bce']:
        criterion = nn.BCEWithLogitsLoss(reduction='none').cuda(opt['g'])
    else:
        criterion = nn.CrossEntropyLoss(reduction='none').cuda(opt['g'])

    optimizer = torch.optim.SGD(model.parameters(), opt['lr'],
                                momentum=opt['momentum'],
                                nesterov=opt['nesterov'],
                                weight_decay=opt['wd'])

    ctx.optimizer = optimizer
    ctx.model = model

    # optionally resume from a checkpoint
    if opt['resume']:
        if os.path.isfile(opt['resume']):
            print("=> loading checkpoint '{}'".format(opt['resume']))
            checkpoint = torch.load(opt['resume'])
            opt['start_epoch'] = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt['resume']))

    cudnn.benchmark = True

    # Data loading code
    train_loader, clean_train_loader, val_loader, weights_loader, train_length = load_data(opt=opt)
    weights_iter = iter(weights_loader)
    ctx.train_loader = train_loader

    if opt['dataset'] == 'celeba':
        df = pd.read_csv('/home/aquarium/celebA/celeba_test_all_attributes.csv')
        indices_a = df[df['Male'] == 1].index.tolist()
        indices_b = df[df['Male'] == 0].index.tolist()
        ctx.indices_a = indices_a
        ctx.indices_b = indices_b

    complete_outputs = torch.ones(train_length).cuda(opt['g'])
    
    if opt['adjust_classes']:
        ctx.class_count = torch.zeros(1000).cuda(opt['g'])
    ctx.max_class_count = 0

    count = torch.zeros_like(complete_outputs).cuda(opt['g'])
    ctx.complete_outputs = complete_outputs
    ctx.count = count
    ctx.max_count = 0
    

    if opt['evaluate']:
        validate(val_loader, train_loader, model, criterion, opt)
        if opt['use_train_clean']:
            train_clean(clean_train_loader, train_loader, model, criterion, opt)
        return

    if ctx.opt['smart_init_sampler']:
        model.eval()
        for i, (x,y,idx) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            out, _ = model(x)
            if ctx.opt['sampler'] == 'alternate':
                S_prob = compute_weights_alt(criterion, weights_loader, model, ctx.opt)
            else:
                S_prob = compute_weights(criterion, weights_loader, model, ctx.opt)
            ctx.S_prob = S_prob
            train_loader.sampler.weights = ctx.S_prob
        model.train()

    for epoch in range(opt['start_epoch'], opt['epochs']):
        ctx.epoch = epoch
        adjust_learning_rate(epoch)
        adjust_temperature(epoch, opt)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, opt, weights_iter, weights_loader)
        # filter out samples too hard to fit
        if ctx.epoch == ctx.opt['ep_drop']:
            srt_idx = list(np.argsort(ctx.count.cpu().numpy()))
            ctx.largest = srt_idx[-ctx.opt['num_drop']:] 
            ctx.S_prob[ctx.largest] = 0
        # evaluate on validation set
        metrics = validate(val_loader, train_loader, model, criterion, opt)
        # evaluate on training set
        if opt['use_train_clean']:
            train_clean(clean_train_loader, train_loader, model, criterion, opt)
        if ctx.opt['sampler'] == 'alternate':
            S_prob = compute_weights_alt(criterion, weights_loader, model, ctx.opt)
        else:
        	S_prob = compute_weights(criterion, weights_loader, model, ctx.opt)
        train_loader.sampler.weights = S_prob

            
        # update stats of the weights
        if opt['save_w_dyn'] or opt['clustering']:
            _ = compute_weights_stats(model, criterion, weights_loader, save_stats=True)
        else:
            if opt['pilot']:
                _ = compute_weights_stats(model, criterion, weights_loader, save_stats=False)
            else:
                pass

        # remember best top@1 and save checkpoint
        top1 = metrics['top1']
        is_best = top1 < best_top1
        best_top1 = min(top1, best_top1)

        save_checkpoint({
            'opt': opt,
            'epoch': epoch + 1,
            'arch': opt['arch'],
            'state_dict': model.state_dict(),
            'best_top1': best_top1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        if ctx.opt['save_hist_until_ep'] > 0 and epoch <= ctx.opt['save_hist_until_ep']:
            plt.bar(np.arange(train_length), ctx.count.cpu().numpy(), color='b')
            plt.ylabel('Count')
            plt.title('Sampling frequency epoch {}'.format(epoch))
            hist_dir = os.path.join(opt.get('o'), opt['exp'], opt['filename']) +'/hist/'
            if not os.path.isdir(hist_dir):
                os.makedirs(hist_dir)
            plt.savefig(hist_dir + 'hist_{}.png'.format(epoch))
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

    if not ctx.opt['evaluate'] and (ctx.opt['save_w_dyn'] or ctx.opt['clustering']):
        with open(os.path.join(ctx.inp_w_dir, 'toweights.pkl'), 'wb') as handle:
            pkl.dump(ctx.toweights, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(ctx.inp_w_dir, 'histograms.pkl'), 'wb') as handle:
            pkl.dump(ctx.histograms, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(ctx.inp_w_dir, 'weights_means.pkl'), 'wb') as handle:
            pkl.dump(ctx.sample_mean.cpu().numpy(), handle, protocol=pkl.HIGHEST_PROTOCOL)
        # concatenate weights vectors
        weights_all_epochs = []
        for i in range(1, ctx.counter + 1):
            name = 'weights_' + str(i) + '.pkl'
            weights_all_epochs.append(np.load(os.path.join(ctx.inp_w_dir, name), allow_pickle=True))
        weights_all_epochs = np.vstack(weights_all_epochs)
        weights_all_epochs = np.transpose(weights_all_epochs)
        with open(os.path.join(ctx.inp_w_dir, 'weights_all_epochs.pkl'), 'wb') as handle:
            pkl.dump(weights_all_epochs, handle, protocol=pkl.HIGHEST_PROTOCOL)
        #shutil.rmtree(os.path.join(ctx.inp_w_dir, 'tmp'))

    if ctx.opt['pilot']:
        if ctx.opt['clustering']:
            pilot = {'weights_all_epochs': weights_all_epochs, 'pilot_directory': ctx.opt['filename']}
            pilot_fn = 'clustering_pilot_' + ctx.opt['dataset'] + '_' + ctx.opt['arch']
        else:
            sorted_w, sorted_idx = torch.sort(ctx.sample_mean, descending=True) 
            pilot = {'sorted_idx': sorted_idx.cpu().numpy(), 'sorted_w': sorted_w.cpu().numpy(), 
                     'pilot_directory': ctx.opt['filename'], 'pilot_saved': ctx.opt['save']}
            pilot_fn = 'pilot_' + ctx.opt['dataset'] + '_' + ctx.opt['arch']
        with open(os.path.join(ctx.opt['o'], 'pilots', pilot_fn + '.pkl'), 'wb') as handle:
            pkl.dump(pilot, handle, protocol=pkl.HIGHEST_PROTOCOL)
