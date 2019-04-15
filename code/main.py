import argparse
import os
import random
import shutil
import time
import warnings
import sys

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
from loader import data_ingredient, load_data 
from sacred import Experiment
import threading
from hook import *
import pickle as pkl

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
    arch = 'allcnn'
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
    sampler = 'default' # (default | our | ufoym )
    wufreq = 1 #weights sampler frequency
    topkw = 500 # number of weight to analyse (default 500)

best_top1 = 0

# for some reason, the db must me created in the global scope
# that is there is not chance to select it easily (fix it)
# if ex.configurations[0]()['dbl']:
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
#print('Creating database')
#ctx.ex.observers.append(MongoObserver.create())
ctx.ex.captured_out_filter = apply_backspaces_and_linefeeds

@data_ingredient.capture
def init(name):
    ctx.global_iters = 0
    ctx.epoch = 0
    ctx.opt = init_opt(ctx)
    if ctx.opt.get('filename', None) is None:
        build_filename(ctx)

    ctx.opt['dataset'] = name
    ctx.metrics = dict()
    ctx.metrics['best_top1'] = best_top1
    ctx.hooks = None
    ctx.top_weights = {'indices': [], 'values': []}
    ctx.init = 0
    # if ctx.opt['sampler'] == 'our':
    #     ctx.weights_logger = create_basic_logger(ctx, 'statistics', idx=0)
    register_hooks(ctx)
    


def cum_abs_diff(cumsum, x_prev, x_new):
    return cumsum + torch.abs(x_prev - x_new)

def compute_weights(model, weights_loader):
    opt = ctx.opt
    model.eval()
    weights = []
    target_list = []
    hist_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(weights_loader):
            data, target = data.cuda(opt['g']), target.cuda(opt['g'])
            output = model(data)
            output = output.squeeze()
            softmax = F.softmax(output, dim=1).cpu().numpy()
            row = np.arange(len(target))
            col = target.cpu().numpy()
            weights.append(np.ones_like(softmax[row, col]) - softmax[row, col])
            target_list.append(target)

    targets_tensor = torch.cat(target_list)
    weights = np.hstack(weights).squeeze()
    weights = torch.FloatTensor(weights)
    weights = weights.cuda(opt['g'])
    sorted_, indices = torch.sort(weights)
    topk = ctx.opt['topkw']
    topk_idx = indices[:topk]
    topk_value = sorted_[:topk]
    num_classes = output.shape[1]
    ctx.top_weights['indices'].append(topk_idx.data.cpu().numpy())
    ctx.top_weights['values'].append(topk_value.data.cpu().numpy())
    
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
    
    inp_w_dir = os.path.join(opt.get('o'), opt['filename']) +'/weights_dir/'
    ctx.inp_w_dir = inp_w_dir
    if ctx.counter == 0:  
        if os.path.exists(inp_w_dir) and os.path.isdir(inp_w_dir):
            shutil.rmtree(inp_w_dir)
        os.makedirs(inp_w_dir)
        os.makedirs(os.path.join(inp_w_dir, 'tmp'))
        
        # initialize sample mean of the weights
        ctx.sample_mean = torch.zeros([1, len(weights_loader.dataset)]).cuda(opt['g'])     
        # here will be stored weights of the last update
        ctx.old_weights = torch.zeros([1, len(weights_loader.dataset)]).cuda(opt['g'])
        ctx.cum_sum_diff = torch.zeros_like(ctx.old_weights).cuda(opt['g'])   
        ctx.cum_sum = 0

    ctx.counter += 1
    ctx.weights = weights
    ctx.sample_mean = ((ctx.counter -1) / ctx.counter) * ctx.sample_mean + (weights / ctx.counter)
    difference = weights - ctx.old_weights
    ctx.cum_sum_diff = cum_abs_diff(ctx.cum_sum, ctx.old_weights, weights)
    ctx.cum_sum += torch.abs(weights)

    with open(os.path.join(inp_w_dir, 'tmp', 'weights_differences_' + str(ctx.counter) + '.pickle'), 'wb') as handle:
        pkl.dump(difference.cpu().numpy(), handle, protocol=pkl.HIGHEST_PROTOCOL)
    ctx.old_weights = weights

    return weights

@batch_hook(ctx, mode='train')
def runner(input, target, model, criterion, optimizer):
    # compute output
        output = model(input)
        loss = criterion(output, target)
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
        #ctx.images = input
        return avg_stats

@epoch_hook(ctx, mode='train')
def train(train_loader, model, criterion, optimizer, epoch, opt):
    data_time = TimeMeter(unit=1)
    ctx.losses = AverageValueMeter()
    ctx.errors = ClassErrorMeter(topk=[1,5])
    n_iters = int(len(train_loader) * opt['wufreq'])

    # switch to train mode
    model.train()

    # end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # tmp var (for convenience)

        # if ctx.global_iters % n_iters == 0:
        #     if opt['sampler'] == 'our':
        #         new_weights = compute_weights(model, weights_loader)
        #         train_loader.sampler.weights = new_weights
        #     else:
        #         # compute dummy weights for visualization
        #         _ = compute_weights(model, weights_loader)

        ctx.i = i
        ctx.global_iters += 1
        input = input.cuda(opt['g'])
        target = target.cuda(opt['g'])
        stats = runner(input, target, model, criterion, optimizer)

        loss = stats['loss']
        top1 = stats['top1']


        if i % opt['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {time:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Err@1 {top1:.3f}\t'.format(
                   epoch, i, len(train_loader),
                   time=data_time.value(), loss=loss, 
                   top1=top1))

    return stats
 
@epoch_hook(ctx, mode='val')
def validate(val_loader, model, criterion, opt):
    data_time = TimeMeter(unit=1)
    losses = AverageValueMeter()
    errors = ClassErrorMeter(topk=[1,5])
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(opt['g'])
            target = target.cuda(opt['g'])

            # compute output
            output = model(input)
            loss = criterion(output, target)

            errors.add(output, target)
            losses.add(loss.item())
         
            loss = losses.value()[0]
            top1 = errors.value()[0]

            # if i % opt['print_freq'] == 0:
            #     print('[{0}/{1}]\t'
            #           'Time {time:.3f}\t'
            #           'Loss {loss:.4f}\t'
            #           'Err@1 {top1:.3f}\t'
            #           'Err@5 {top5:.3f}'.format(
            #            i, 
            #            len(val_loader),
            #            time=data_time.value(), loss=loss, 
            #            top1=top1, top5=top5))

        print(' * Err@1 {top1:.3f}'
              .format(top1=top1))
    stats = {'loss': loss, 'top1': top1}
    ctx.metrics = stats
    ctx.images = input
    return stats


@epoch_hook(ctx, mode='train_clean')
def clean_train(val_loader, model, criterion, opt):
    data_time = TimeMeter(unit=1)
    losses = AverageValueMeter()
    errors = ClassErrorMeter(topk=[1,5])
    # switch to evaluate mode
    #model.eval()
    model.train()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(opt['g'], non_blocking=True)
            target = target.cuda(opt['g'], non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            errors.add(output, target)
            losses.add(loss.item())
         
            loss = losses.value()[0]
            top1 = errors.value()[0]

            # if i % opt['print_freq'] == 0:
            #     print('[{0}/{1}]\t'
            #           'Time {time:.3f}\t'
            #           'Loss {loss:.4f}\t'
            #           'Err@1 {top1:.3f}\t'
            #           'Err@5 {top5:.3f}'.format(
            #            i, 
            #            len(val_loader),
            #            time=data_time.value(), loss=loss, 
            #            top1=top1, top5=top5))

        print(' * Err@1 {top1:.3f}'
              .format(top1=top1))
    stats = {'loss': loss, 'top1': top1}
    ctx.metrics = stats
    ctx.images = input
    return stats

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not ctx.opt['save']:
        return
    opt = ctx.opt
    #fn = os.path.join(opt['o'], opt['arch'], opt['filename']) + '.pth.tar'
    fn = os.path.join(opt['o'], opt['filename'], 'last_model.pth.tar')
    r = gitrev(opt)
    meta = dict(SHA=r[0], STATUS=r[1], DIFF=r[2])
    state.update({'meta': meta})
    torch.save(state, fn)
    if is_best:
        # filename = os.path.join(opt['o'], opt['arch'], 
        #                     opt['filename']) + '_best.pth.tar'
        filename = os.path.join(opt['o'], opt['filename'], 
                            'best.pth.tar')
        shutil.copyfile(fn, filename)


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


@train_hook(ctx)
def main_worker(opt):
    global best_top1

     # create model
    if opt['pretrained']:
        print("=> using pre-trained model '{}'".format(opt['arch']))
        if 'allcnn' in opt['arch'] or 'wrn' in opt['arch'] or 'lenet' in opt['arch']:
            model = models.__dict__[opt['arch']](opt)
            #load_pretrained(model, opt['dataset'])
            load_pretrained(model, name_ptr)
        else:
            model = models.__dict__[opt['arch']](opt, pretrained=True)
    else:
        print("=> creating model '{}'".format(opt['arch']))
        model = models.__dict__[opt['arch']](opt)

    if opt['ng'] == 0:
        torch.cuda.set_device(opt['g'])
        model = model.cuda(opt['g'])
    else:
        model = torch.nn.DataParallel(model, 
                        device_ids=range(opt['g'], opt['g'] + opt['ng'],
                        output_device=opt['g'])).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(opt['g'])

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
    train_loader, val_loader, clean_train_loader, weights_loader = load_data(opt=opt)
    ctx.train_loader = train_loader

    ctx.counter = 0     # count the number of times weights are updated

    if opt['evaluate']:
        #clean_train(clean_train_loader, model, criterion, opt)
        validate(val_loader, model, criterion, opt)
        return

    for epoch in range(opt['start_epoch'], opt['epochs']):
        ctx.epoch = epoch
        adjust_learning_rate(epoch)

        if opt['sampler'] == 'our':
            new_weights = compute_weights(model, weights_loader)
            train_loader.sampler.weights = new_weights
        else:
            # compute dummy weights for visualization
            _ = compute_weights(model, weights_loader)
            
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, opt)

        clean_train(clean_train_loader, model, criterion, opt)
        # evaluate on validation set
        metrics = validate(val_loader, model, criterion, opt)

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


    with open(os.path.join(ctx.inp_w_dir, 'top_weights.pickle'), 'wb') as handle:
        pkl.dump(ctx.top_weights, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(ctx.inp_w_dir, 'histograms.pickle'), 'wb') as handle:
        pkl.dump(ctx.histograms, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(ctx.inp_w_dir, 'weights_means.pickle'), 'wb') as handle:
        pkl.dump(ctx.sample_mean.cpu().numpy(), handle, protocol=pkl.HIGHEST_PROTOCOL)
    # concatenate weights differences
    weights_diff = []
    for i in range(1, ctx.counter + 1):
        name = 'weights_differences_' + str(i) + '.pickle'
        weights_diff.append(np.load(os.path.join(ctx.inp_w_dir, 'tmp', name)))
    weights_diff = np.vstack(weights_diff)
    with open(os.path.join(ctx.inp_w_dir, 'weights_differences.pickle'), 'wb') as handle:
        pkl.dump(weights_diff, handle, protocol=pkl.HIGHEST_PROTOCOL)
    shutil.rmtree(os.path.join(ctx.inp_w_dir, 'tmp'))

