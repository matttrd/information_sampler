import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchnet.meter import ClassErrorMeter, ConfusionMeter, TimeMeter, AverageValueMeter
#import torchvision.models as models
import models
from loader import data_ingredient, load_data 
from sacred import Experiment
import threading
from hook import *

# local thread used as a global context
ctx = threading.local()
ex = Experiment('base', ingredients=[data_ingredient])
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
    # seed
    seed = None
    # gpu index (or starting gpu idx)
    g = int(0)
    # number of gpus for data parallel
    ng = int(0)
    # learning schedule
    lrs = '[[0,0.1],[60,0.02],[120,0.004],[160,0.0008],[180,0.0001]]'
    # dropout (if any)
    d = 0.
    # file logger
    fl = False
    # Tensorflow logger
    tfl = False
    dbl = False
    # output dir
    o = '../results/'
    #save model
    save = False
    #whitelist for filename
    whitelist = '[]'
    # marker
    marker = ''


best_top1 = 0

# for some reason, the db must me created in the global scope
if ex.configurations[0]()['dbl']:
    from sacred.observers import MongoObserver
    from sacred.utils import apply_backspaces_and_linefeeds
    print('Creating database')
    ctx.ex.observers.append(MongoObserver.create())
    ctx.ex.captured_out_filter = apply_backspaces_and_linefeeds

@data_ingredient.capture
def init(name):
    ctx.epoch = 0
    ctx.opt = init_opt(ctx)
    ctx.opt['dataset'] = name
    ctx.metrics = dict()
    ctx.metrics['best_top1'] = best_top1
    ctx.hooks = None
    register_hooks(ctx)
    

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
                 'top1': ctx.errors.value()[0], 
                 'top5': ctx.errors.value()[1]}
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
    # switch to train mode
    model.train()

    # end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # tmp var (for convenience)
        ctx.i = i
        input = input.cuda(opt['g'], non_blocking=True)
        target = target.cuda(opt['g'], non_blocking=True)
        stats = runner(input, target, model, criterion, optimizer)

        loss = stats['loss']
        top1 = stats['top1']
        top5 = stats['top5']

        if i % opt['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {time:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Err@1 {top1:.3f}\t'
                  'Err@5 {top5:.3f}'.format(
                   epoch, i, len(train_loader),
                   time=data_time.value(), loss=loss, 
                   top1=top1, top5=top5))

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
            input = input.cuda(opt['g'], non_blocking=True)
            target = target.cuda(opt['g'], non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            errors.add(output, target)
            losses.add(loss.item())
         
            loss = losses.value()[0]
            top1 = errors.value()[0]
            top5 = errors.value()[1]

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

        print(' * Err@1 {top1:.3f} Err@5 {top5:.3f}'
              .format(top1=top1, top5=top5))
    stats = {'loss': loss, 'top1': top1, 'top5': top5}
    ctx.metrics = stats
    ctx.images = input
    return stats


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not ctx.opt['save']:
        return
    opt = ctx.opt
    fn = os.path.join(opt['o'], opt['arch'], opt['filename']) + '.pth.tar'
    r = gitrev(opt)
    meta = dict(SHA=r[0], STATUS=r[1], DIFF=r[2])
    state.update({'meta': meta})
    th.save(state, fn)
    if is_best:
        filename = os.path.join(opt['o'], opt['arch'], 
                            opt['filename']) + '_best.pth.tar'
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
        if 'allcnn' in opt['arch'] or 'wrn' in opt['arch']:
            model = models.__dict__[opt['arch']](opt)
            load_pretrained(model, opt['dataset'])
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
    train_loader, val_loader = load_data(opt=opt)

    if opt['evaluate']:
        validate(val_loader, model, criterion, opt)
        return

    for epoch in range(opt['start_epoch'], opt['epochs']):
        ctx.epoch = epoch
        adjust_learning_rate(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, opt)

        # evaluate on validation set
        metrics = validate(val_loader, model, criterion, opt)

        # remember best top@1 and save checkpoint
        top1 = metrics['top1']
        is_best = top1 < best_top1
        best_top1 = min(top1, best_top1)

        save_checkpoint({
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