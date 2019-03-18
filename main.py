from __future__ import print_function
import argparse
import random
import copy
import numpy as np
np.set_printoptions(threshold=np.nan)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.metrics as sm
import threading
import pickle as pkl
from sampler import ImbalancedDatasetSampler


context = threading.local()
context.top_weights = {'indices': [], 'values': []}
context.init = 0

def compute_weights(model, weights_loader, device):
    model.eval()
    weights = []
    target_list = []
    hist_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(weights_loader):
            data, target = data.to(device), target.to(device)
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
    weights = weights.to(device)
    sorted_, indices = torch.sort(weights)
    topk = context.args.topk
    topk_idx = indices[:topk]
    topk_value = sorted_[:topk]
    num_classes = output.shape[1]
    context.top_weights['indices'].append(topk_idx.data.cpu().numpy())
    context.top_weights['values'].append(topk_value.data.cpu().numpy())
    if context.init == 0:
        context.histograms = {str(k): [] for k in range(num_classes)}
        context.histograms['total'] = []
        context.init = 1
    for cl in range(num_classes):
        idx_w_cl = targets_tensor == cl
        w_cl = weights[idx_w_cl]
        hist, bin_edges = np.histogram(w_cl.cpu().numpy(), bins=100, range=(0,1))
        context.histograms[str(cl)].append(hist)
    hist, bin_edges = np.histogram(weights.cpu().numpy(), bins=100, range=(0,1))
    context.histograms['total'].append(hist)
    context.histograms['bin_edges'] = bin_edges
    return weights
    
def vis(test_accs, confusion_mtxes, labels, sampler, figsize=(20, 8)):
    cm = confusion_mtxes[np.argmax(test_accs)]
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    # accuracy
    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(test_accs, 'g')
    plt.title('Best test accuracy = %2.2f%%' % max(test_accs))
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.grid(True)
    # conf matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.savefig('./results_' + str(sampler) + '_sampler.png')

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = torch.nn.Dropout2d()
#         self.fc1 = torch.nn.Linear(320, 50)
#         self.fc2 = torch.nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x


class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)

class Net(torch.nn.Module):
    def __init__(self, opt, c1=96, c2= 192):
        super(allcnn, self).__init__()
        self.name = 'allcnn'

        if opt['dataset'] == 'cifar10' or opt['dataset'] == 'cifar100':
            in_ch = 3
            out_ch = 8
        elif opt['dataset'] == 'mnist':
            in_ch = 1
            out_ch = 7
        if (not 'l2' in opt) or opt['l2'] < 0:
            opt['l2'] = 1e-3

        if (not 'd' in opt) or opt['d'] < 0:
                opt['d'] = 0.5

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.BatchNorm2d(co),
                nn.ReLU(True))

        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(in_ch,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,num_classes,1,1),
            nn.AvgPool2d(out_ch),
            View(num_classes))    

def train(args, model, device, train_loader, weights_loader, optimizer, loss_fun, epoch):
    model.train()
    n_iters = int(len(train_loader) * args.freq)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if batch_idx % n_iters == 0:
            print('recomputing weights')
            if args.sampler == 'our':
                new_weights = compute_weights(model, weights_loader, device)
                train_loader.sampler.weights = new_weights
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()
        
def test(args, model, device, test_loader):
    model.eval()
    correct = 0
    targets, preds = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            targets += list(target.cpu().numpy())
            preds += list(pred.cpu().numpy())
    test_acc = 100. * correct / len(test_loader.dataset)
    confusion_mtx = sm.confusion_matrix(targets, preds)
    return test_acc, confusion_mtx
        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--sampler', type=str, default='our', 
                        help='sampler for training (default: "our")')
    parser.add_argument('--freq', type=float, default=0.1, 
                        help='weights sampler frequency')
    parser.add_argument('--topk', type=int, default=500, metavar='N',
                        help='number of weight to analyse (default 500)')                        
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    context.args = args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    with open('./imbalanced_dataset.pickle', 'rb') as handle:
        data = pkl.load(handle)
    imbalanced_train_dataset = data['imbalanced_train_dataset']
    num_classes = data['num_classes']
    test_loader = data['test_loader']
    class_labels = range(num_classes)
    
    # our sampler
    dataset_length = len(imbalanced_train_dataset)
    weights_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=1024, shuffle=False, **kwargs) # used for the computation of the weights
    weights_init = torch.FloatTensor(np.random.uniform(0, 1, dataset_length))
    weights_init = weights_init.to(device)
    our_sampler = torch.utils.data.WeightedRandomSampler(weights=weights_init, num_samples=int(len(weights_init)), replacement=True)
    # ufoym sampler
    ufoym_sampler = ImbalancedDatasetSampler(imbalanced_train_dataset)
    
    # data loader
    if args.sampler == 'our':
        imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=args.batch_size, sampler=our_sampler, **kwargs)
    elif args.sampler == 'ufoym':
        imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=args.batch_size, sampler=ufoym_sampler, **kwargs)
    elif args.sampler == 'default':
        imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=args.batch_size, **kwargs)
    
    # define model
    opt = dict()
    opt['dataset'] = 'mnist'
    model = Net(opt).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # main loop
    test_accs, confusion_mtxes = [], []    
    for epoch in range(1, args.epochs + 1):
        print('Train Epoch: {}'.format(epoch))
        # if args.sampler == 'our':
        #     new_weights = compute_weights(model, weights_loader, device)
        #     imbalanced_train_loader.sampler.weights = new_weights
        train(args, model, device, imbalanced_train_loader, weights_loader, optimizer, loss, epoch)
        test_acc, confusion_mtx = test(args, model, device, test_loader)
        test_accs.append(test_acc)
        confusion_mtxes.append(confusion_mtx)
        print('Best test acc = %2.2f%%' % max(test_accs), end='\n', flush=True)
    
    if args.sampler == 'our':
        with open('./top_weights_' + str(args.sampler) + '_sampler.pickle', 'wb') as handle:
            pkl.dump(context.top_weights, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open('./histograms_' + str(args.sampler) + '_sampler.pickle', 'wb') as handle:
            pkl.dump(context.histograms, handle, protocol=pkl.HIGHEST_PROTOCOL)
      
    vis(test_accs, confusion_mtxes, class_labels, args.sampler)
    
    if (args.save_model):
        torch.save(model.state_dict(),'mnist_cnn_' + str(args.sampler) + '_sampler.pt')
        
if __name__ == '__main__':
    main()