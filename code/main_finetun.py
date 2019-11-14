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
from utils_lt import shot_acc, shot_acc_cifar
from IPython import embed
import matplotlib.pyplot as plt
import defaults
from scipy import interpolate

# local thread used as a global context
ctx = threading.local()
ex = Experiment('information sampler', ingredients=[data_ingredient])
ctx.ex = ex


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
    forgetting_stats = True
    cifar_imb_factor = None
    focal_loss = False
    gamma = 1.
    lb = False # linear warm up

    x_0 = np.log(2)
    x_1 =  np.log(10/2)
    beta_0 = 0.1
    beta_1 = beta_0

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
    if ctx.opt['cifar_imb_factor'] is not None:
        ctx.acc_per_class = []
    # if ctx.opt['sampler'] == 'our':
    #     ctx.weights_logger = create_basic_logger(ctx, 'statistics', idx=0)
    register_hooks(ctx)



def cum_abs_diff(cumsum, x_prev, x_new):
    return cumsum + torch.abs(x_prev - x_new)

def compute_weights(complete_outputs, outputs, targets, idx, criterion):
    output = outputs.detach()
    for i, index in enumerate(idx):
        o = output[i].reshape(-1, output.shape[1])
        if ctx.opt['bce']:
            new_targets = logical_index(targets, output.shape).float()
        else:
            new_targets = targets

        complete_outputs[index] = criterion(o, new_targets[i].unsqueeze(0)).mean()

        ctx.count[index] += 1
        if ctx.opt['adjust_classes']:
            ctx.class_count[targets[i]] += 1
            max_classes = ctx.class_count.max()
            if max_classes > ctx.max_class_count:
                ctx.max_class_count = max_classes

    max_ = ctx.count[idx].max()
    if max_ > ctx.max_count:
            ctx.max_count = max_

    complete_losses = complete_outputs
    ncounts = ctx.count / ctx.max_count
    if ctx.opt['dyncount']:
        temp = ctx.opt['temperature'] * ncounts
    else:
        temp = ctx.opt['temperature']

    if ctx.opt['adjust_classes']:
        for i, index in enumerate(idx):
            ratio = ctx.opt['ac_scaler'] * ctx.class_count[targets[i]] / ctx.max_class_count
            complete_losses[index] = complete_losses[index] / ratio

    nrm = temp * complete_losses.max() if ctx.opt['normalizer'] else temp
    if ctx.opt['sampler'] == 'tunnel':
        S_prob = torch.exp(-complete_losses / nrm)
    else:
        S_prob = 1 - torch.exp(-complete_losses / nrm)


    ####        ZANCA'S METHOD      ###
    #classes = ctx.classes
    ## Updating losses in the list of dictionaries
    #for j, ind in enumerate(idx):
    #    classes[targets[j].item()][ind.item()] = complete_outputs[ind.item()].item()
    #F_min = (ctx.opt['x_0'] + ctx.opt['x_1'])/ 2 # torch.log(2), torch.log(10/2)
    ## Compute medians per class
    #median = torch.zeros(10)
    #for class_index in range(len(classes)):
    #    losses_median = []
    #    for _, los in classes[class_index].items():
    #        losses_median.append(los)
    #    median[class_index] = torch.median(torch.tensor(losses_median))
    #medians = ctx.medians
    #for cla, dict_loss in enumerate(classes):
    #    start = time.time()
    #    #ct = 0
    #    for indeces in dict_loss:
    #        medians[indeces] = median[cla]
    #        #ct += 1
    ## print(len(dict_loss) if type(dict_loss) is not float else 1)  
    #complete_loss_normalized = (complete_losses - medians - torch.tensor(F_min)) / medians
    #J = F(complete_loss_normalized, ctx.opt['x_0'], ctx.opt['x_1'], ctx.opt['beta_0'], ctx.opt['beta_1'])
    #S_prob = (1- torch.exp(-J)) * medians / ctx.complete_cardinality

    return S_prob

def F(loss, x_0, x_1, beta_0, beta_1):
    return torch.exp((loss - x_0) / beta_0) +  torch.exp(-(loss - x_1)/ beta_1)



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
    num_classes = output.shape[1]


    if ctx.counter == 0:
        if save_stats:
            if opt['pilot']:
                inp_w_dir = os.path.join(opt.get('o'), 'pilots', opt['filename']) +f'{os.sep}input_weights{os.sep}'
            else:
                inp_w_dir = os.path.join(opt.get('o'), opt['exp'], opt['filename']) + f'{os.sep}input_weights{os.sep}'
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

def updating_forgetting_stats(loss, output, target, idx, len_dataset):
    # Creating the dicitonary that is keeping the statiscs
    if not ctx.forgetting_stats:
        ctx.forgetting_stats = {}
        for i in range(len_dataset):
            # ctx.forgetting_stats[i] = {'loss': [], 'acc': [], 'margin': [], 'counts': []} #if you want to get more stats
            ctx.forgetting_stats[i] = {'loss': [], 'acc': [], 'margin': []}

    # Updating the statiscs and loss for the forgetting experiment
    _, predicted = torch.max(output.data, 1)
    acc = predicted == target
    for i, index in enumerate(idx):
        # Compute the correct class
        output_correct_class = output.data[i, target[i].item()]
        sorted_output, _ = torch.sort(output.data[i, :])
        if acc[i]:
            output_highest_incorrect_class = sorted_output[-2]
        else:
            # Example misclassified, highest incorrect class is max output
            output_highest_incorrect_class = sorted_output[-1]
        margin = output_correct_class.item() - output_highest_incorrect_class.item()

        ctx.forgetting_stats[index.item()]['loss'].append(loss[i].item())
        ctx.forgetting_stats[index.item()]['acc'].append(acc[i].sum().item()) #.sum() to habe number
        ctx.forgetting_stats[index.item()]['margin'].append(margin)
        # ctx.forgetting_stats[index.item()]['counts'].append(ctx.count[index.item()].item())


@batch_hook(ctx, mode='train')
def runner(input, target, model, criterion, optimizer, idx, complete_outputs):
    # compute output
        # embed()
        output, _ = model(input)

        if ctx.opt['bce']:
            orig_target = target.clone()
            new_target = logical_index(target, output.shape).float()
        else:
            new_target = target

        loss = criterion(output, new_target)

        if ctx.opt['forgetting_stats']:
            updating_forgetting_stats(loss, output, target, idx, complete_outputs.shape[0])

        loss = loss.mean()

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
        S_prob = compute_weights(complete_outputs, output, target, idx, criterion)
        avg_stats['top1'] = (100 - avg_stats['top1']) / 100.

        return avg_stats, S_prob

@epoch_hook(ctx, mode='train')
def train(train_loader, model, criterion, optimizer, epoch, opt, complete_outputs):
    data_time = TimeMeter(unit=1)
    ctx.losses = AverageValueMeter()
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
        stats, S_prob = runner(input, target, model, criterion, optimizer, idx, complete_outputs)
        ctx.S_prob = S_prob
        if opt['sampler'] == 'invtunnel' or opt['sampler'] == 'tunnel':
            train_loader.sampler.weights = ctx.S_prob

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
    errors  = ClassErrorMeter(topk=[1,5])
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

    if '_lt' in ctx.opt['dataset'] or ctx.opt['cifar_imb_factor'] is not None:
        if ctx.opt['dataset'] == 'imagenet_lt':
            many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds, targets, train_dataset)
            stats['many_acc_top1'] = many_acc_top1
            stats['median_acc_top1'] = median_acc_top1
            stats['low_acc_top1'] = low_acc_top1
        else:
            acc = shot_acc_cifar(preds, targets, train_dataset)
            ctx.acc_per_class.append(acc)

    ctx.metrics = stats
    return stats

@epoch_hook(ctx, mode='train_clean')
def train_clean(loader, train_dataset, model, criterion, opt):
    losses = AverageValueMeter()
    errors = ClassErrorMeter(topk=[1,5])
    # switch to evaluate mode
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for i, (input, target, idx) in tqdm(enumerate(loader)):
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
            S_prob = compute_weights(ctx.complete_outputs, output, target, idx, criterion)
            if opt['sampler'] == 'invtunnel' or opt['sampler'] == 'tunnel':
                ctx.train_loader.sampler.weights = S_prob

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

    # if ctx.opt['dataset'] == 'imagenet_lt':
    #     many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds, targets, train_dataset)
    #     stats['many_acc_top1'] = many_acc_top1
    #     stats['median_acc_top1'] = median_acc_top1
    #     stats['low_acc_top1'] = low_acc_top1

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
    if len(opt['save_epochs'])>0 and ctx.epoch in opt['save_epochs']:
        if opt['pilot']:
            fn = os.path.join(opt['o'], 'pilots', opt['filename'], 'model_' + str(ctx.epoch) + '.pth.tar')
        else:
            fn = os.path.join(opt['o'], opt['exp'], opt['filename'], 'model_' + str(ctx.epoch) + '.pth.tar')
        torch.save(state, fn)

    if len(ctx.opt['save_counts_list']) > 0 and ctx.epoch in ctx.opt['save_counts_list'] or \
            (ctx.opt['freq_save_counts'] > 0 and ctx.epoch == ctx.opt['epochs']-1):
            counts_dir = os.path.join(opt.get('o'), opt['exp'], opt['filename']) + f'{os.sep}sample_counts{os.sep}'
            if not os.path.isdir(counts_dir):
                os.makedirs(counts_dir)
            with open(os.path.join(counts_dir, 'sample_counts_' + str(ctx.epoch) + '.pkl'), 'wb') as handle:
                pkl.dump(ctx.count.cpu().numpy(), handle, protocol=pkl.HIGHEST_PROTOCOL)

            weights_dir = os.path.join(opt.get('o'), opt['exp'], opt['filename']) + f'{os.sep}weigths_folder{os.sep}'
            os.makedirs(weights_dir, exist_ok=True)
            with open(os.path.join(weights_dir, 'S_weights_' + str(ctx.epoch) + '.pkl'), 'wb') as handle:
                pkl.dump(ctx.S_prob.cpu().numpy(), handle, protocol=pkl.HIGHEST_PROTOCOL)
            # with open(os.path.join(ctx.inp_w_dir, 'weights_means.pkl'), 'wb') as handle:
            # pkl.dump(ctx.sample_mean.cpu().numpy(), handle, protocol=pkl.HIGHEST_PROTOCOL)
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
    # if len(opt['temperatures']) > 0:
    #     temps = opt['temperatures']
    #     initial_temp = temps[0]
    #     final_temp = temps[1]
    #     ratio = (epoch / opt['epochs']) ** 0.5
    #     ctx.opt['temperature'] = final_temp * ratio + initial_temp * (1 - ratio)
    if opt['temperatures'] != '':
        opt['temperature'] = schedule(ctx, k='temperature')
    return

def get_lb_warmup(e):
    if isinstance(ctx.opt['lrs'], str):
        lrs = np.array(json.loads(ctx.opt['lrs']))
    else:
        lrs = np.array(ctx.opt['lrs'])
    interp = interpolate.interp1d(lrs[:,0], lrs[:,1], kind='linear',fill_value='extrapolate')
    lr = np.asscalar(interp(e))
    return lr

@train_hook(ctx)
def main_worker(opt):
    global best_top1
    model = create_and_load_model(ctx.opt)
    if opt['ng'] == 0:
        torch.cuda.set_device(opt['g'])
        model = model.cuda(opt['g'])
    else:
        model = torch.nn.DataParallel(model,
                        device_ids=range(opt['g'], opt['g'] + opt['ng']),
                        output_device=opt['g']).cuda()

    # embed()

    # define loss function (criterion) and optimizer
    if opt['bce']:
        criterion = nn.BCEWithLogitsLoss(reduction='none').cuda(opt['g'])
    elif opt['focal_loss']:
        from losses import FocalLoss
        criterion = FocalLoss(gamma=opt['gamma'])
    else:
        criterion = nn.CrossEntropyLoss(reduction='none').cuda(opt['g'])

    # optionally resume from a checkpoint
    if opt['resume']:
        if os.path.isfile(opt['resume']):
            print("=> loading checkpoint '{}'".format(opt['resume']))
            checkpoint = torch.load(opt['resume'])
            opt['start_epoch'] = 0
            best_top1 = checkpoint['best_top1']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt['resume']))


    classifier = model.linear
    import math
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, math.sqrt(2./m.in_features))
            m.bias.data.fill_(-np.log(get_num_classes(opt) - 1))

    feats_net_pars = list(model.conv1.parameters()) + list(model.bn1.parameters())
    feats_net_pars+= list(model.layer1.parameters())
    feats_net_pars+= list(model.layer2.parameters())
    feats_net_pars+= list(model.layer3.parameters())
    feats_net_pars+= list(model.layer4.parameters())

    if opt['adam']:
        optimizer = torch.optim.Adam(model.parameters(), opt['lr'],
                                betas = (0.9,0.99),
                                weight_decay=opt['wd'])
    else:
        # optimizer = torch.optim.SGD(model.parameters(), opt['lr'],
        #                             momentum=opt['momentum'],
        #                             nesterov=opt['nesterov'],
        #                             weight_decay=opt['wd'])
        optimizer = torch.optim.SGD([
                {'params': feats_net_pars, 'lr': 0.},
                {'params': classifier.parameters(), 'lr':1e-3}
            ], lr=opt['lr'], momentum=0.9)
    
    ctx.optimizer = optimizer
    ctx.model = model

    
    cudnn.benchmark = True

    # Data loading code
    train_loader, clean_train_loader, val_loader, weights_loader, train_length = load_data(opt=opt)
    ctx.train_loader = train_loader
    #ctx.counter = 1

    #complete_outputs = torch.ones(train_length).cuda(opt['g'])
    complete_outputs = train_loader.sampler.weights.clone().cuda(opt['g'])

    # # Create a list of dictionaries of indeces and losses
    # print('Classes for modified IS')
    # classes = [{} for i in range(10)]
    # for i, (input, target, index) in enumerate(weights_loader):
    #     for j, ind in enumerate(index):
    #         classes[target[j].item()][ind.item()] = complete_outputs[index[j].item()].item()

    # ctx.classes = classes
    # cardinality = [len(i)  for i in classes]

    # ctx.medians = torch.zeros_like(complete_outputs).cuda(ctx.opt['g'])
    # ctx.complete_cardinality = torch.zeros_like(complete_outputs).cuda(ctx.opt['g'])
    # for i, cla in enumerate(classes):
    #     for ind in cla:
    #         ctx.complete_cardinality[ind] = cardinality[i]



    if opt['adjust_classes']:
        ctx.class_count = torch.ones(get_num_classes(opt)).cuda(opt['g'])
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

        model.train()

    import time
    for epoch in range(opt['start_epoch'], opt['epochs']):
        start_t = time.time()
        ctx.epoch = epoch
        # if not opt['adam']:
        #     if isinstance(opt['lrs'], str):
        #         lrs = np.array(json.loads(opt['lrs']))
        #     else:
        #         lrs = np.array(opt['lrs'])
        #     if opt['lb'] and epoch <= lrs[1,0]:
        #         lr = get_lb_warmup(epoch)
        #         print(lr)

        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = lr
        #     else:
        #         adjust_learning_rate(epoch)

        adjust_temperature(epoch, opt)
        # if opt['sampler'] == 'invtunnel' or opt['sampler'] == 'tunnel':
        #     new_weights = compute_weights_stats(model, criterion, weights_loader)
        #     train_loader.sampler.weights = new_weights
        # else:
        #     # compute dummy weights for visualization
        # if ctx.opt['save']:
        #   _ = compute_weights_stats(model, criterion, weights_loader)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, opt, complete_outputs)
        # filter out samples too hard to fit

        # evaluate on validation set
        metrics = validate(val_loader, train_loader, model, criterion, opt)
        # evaluate on training set
        if opt['use_train_clean']:
            train_clean(clean_train_loader, train_loader, model, criterion, opt)
        end_t = time.time()
        print(end_t-start_t)
        # update stats of the weights
        if opt['pilot']:
            _ = compute_weights_stats(model, criterion, weights_loader, save_stats=False)


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

    # Saving forgetting_stats
    save_dir = os.path.join(ctx.opt.get('o'), ctx.opt['exp'], ctx.opt['filename'], 'forgetting_stats')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'stats.pkl'), 'wb') as handle:
        pkl.dump(ctx.forgetting_stats, handle, protocol=pkl.HIGHEST_PROTOCOL)

    if ctx.opt['cifar_imb_factor'] is not None:
        acc_dir = os.path.join(ctx.opt.get('o'), ctx.opt['exp'], ctx.opt['filename'], 'accuracies_per_class')
        os.makedirs(acc_dir, exist_ok=True)
        with open(os.path.join(acc_dir, 'accuracies.pkl'), 'wb') as handle:
            pkl.dump(np.array(ctx.acc_per_class), handle, protocol=pkl.HIGHEST_PROTOCOL)

    if ctx.opt['pilot']:
        # we need the array of sorted indexes in the form [easy --> hard]
        if ctx.opt['sampler'] == 'tunnel':
            # weights = p
            sorted_w, sorted_idx = torch.sort(ctx.sample_mean, descending=True)
        else:
            # weights = 1-p
            sorted_w, sorted_idx = torch.sort(ctx.sample_mean, descending=False)
        pilot = {'sorted_idx': sorted_idx.cpu().numpy(), 'sorted_w': sorted_w.cpu().numpy(),
                 'pilot_directory': ctx.opt['filename'], 'pilot_saved': ctx.opt['save']}
        pilot_fn = 'pilot_' + ctx.opt['dataset'] + '_' + ctx.opt['arch'] + '_' + ctx.opt['sampler']
        with open(os.path.join(ctx.opt['o'], 'pilots', pilot_fn + '.pkl'), 'wb') as handle:
            pkl.dump(pilot, handle, protocol=pkl.HIGHEST_PROTOCOL)
