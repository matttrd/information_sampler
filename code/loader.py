from __future__ import division, print_function, unicode_literals
from sacred import Ingredient, Experiment
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchnet as tnt
import torch 
from sampler import ImbalancedDatasetSampler
from torch.utils.data import Subset
import numpy as np
import copy
import random
from models import get_num_classes

data_ingredient = Ingredient('dataset')

@data_ingredient.config
def cfg():
    name = 'cifar10'  # dataset filename
    source = '../data/'
    shuffle = True #only for training set by default
    frac = 1 # fraction of dataset used
    norm = False
    perc = 0 # percentage of most exemples to be removed
    mode = 0 #remove: most difficult (0) | easy samples (1) random (2)

class MyDataset(Dataset):
    def __init__(self, data, source, train, download, transform):
        if data == 'cifar10':
            self.data =  datasets.CIFAR10(source, train=train, download=download, transform=transform)
        elif data == 'cifar100':
            self.data =  datasets.CIFAR100(source, train=train, download=download, transform=transform)
        else:
            print('Only CIFAR 10/100 allowed!')

    def __getitem__(self, index):
        data, target = self.data[index][0], self.data[index][1]
        return data, target, index

    def __len__(self):
        return len(self.data)

@data_ingredient.capture
def load_data(name, source, shuffle, frac, perc, mode, norm, opt):

    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))])

        transforms_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))])
    elif name == 'mnist':
        transform_train = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                           ])
        transforms_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    elif name == 'cifar10.1':
        if opt['evaluate']:
            test_dataset = dict()
            data = np.load('../data/cifar10.1_v6_data.npy').transpose((0,3,1,2)) / 255.
            data = np.float32(data)
            labels = np.load('../data/cifar10.1_v6_labels.npy').tolist()
            #print(labels.dtype)
            test_dataset = tnt.dataset.TensorDataset([data, labels])
            test_loader = torch.utils.data.DataLoader(
                        test_dataset, 
                            batch_size=opt['b'], 
                            shuffle=False, 
                            num_workers=opt['j'], 
                            pin_memory=True)
            return None, test_loader, None, None
        else:
            raise ValueError("cifar10.1 can be used only in evaluation mode")
    else:
        raise NotImplementedError
    
    # train_dataset = datasets.__dict__[name.upper()](source, train=True, download=True,
    #                    transform=transform_train)
    # clean_train_dataset = datasets.__dict__[name.upper()](source, train=True, download=True,
    #                    transform=transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
    #              transforms.Normalize((0, 0, 0), (1, 1, 1))]))
    
    # test_dataset = datasets.__dict__[name.upper()](source, train=False, download=True,
    #                    transform=transforms_test)

    train_dataset = MyDataset(name, source, train=True, download=True,
                       transform=transform_train)
    clean_train_dataset = MyDataset(name, source, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))]))
    
    test_dataset = MyDataset(name, source, train=False, download=True,
                       transform=transforms_test)

    train_length = len(train_dataset)
    test_length = len(test_dataset)

    indices = np.arange(0,train_length)

    if perc > 0:
        print('Dataset reduction of ', perc)
        sd_idx = np.squeeze(torch.load('sorted_datasets.pz')[name])
        if mode == 0:
            idx = sd_idx[int((1 - perc) * train_length):]
        elif mode == 1:
            idx = sd_idx[:int(perc * train_length)]
        elif mode == 2:
            sd_idx = np.random.permutation(sd_idx)
            idx = sd_idx[:int(perc * train_length)]
        else:
            raise(ValueError('Valid mode values: 0,1,2'))
        mask = np.ones(train_length, dtype=bool)
        mask[idx] = False
        indices = indices[mask]
        train_dataset = Subset(train_dataset, indices)
        
        # train_dataset.targets = np.array(train_dataset.targets)[mask]
        # train_dataset.data = np.array(train_dataset.data)[mask]
        train_length = len(train_dataset)
        print('New Dataset length is ', train_length)

    if frac < 1:
        indices = indices[:int(frac * train_length)]
        train_dataset = Subset(train_dataset, indices)
        train_length = len(train_dataset)
        # train_dataset.targets = train_dataset.targets[:int(frac * train_length)]
        # train_dataset.data = train_dataset.data[:int(frac * train_length)]
        # test_dataset.targets = test_dataset.targets[:int(frac * test_length)]
        # test_dataset.data = test_dataset.data[:int(frac * test_length)]
        # train_length = len(train_dataset)
        # test_length = len(test_dataset)

    if opt['unbalanced']: 
        num_classes = get_num_classes(opt)
        class_labels = range(num_classes)
        sample_probs = torch.rand(num_classes)
        
        idx_to_del = [i for i, label in enumerate(train_dataset.data.targets) 
                      if random.random() > sample_probs[label]]
        mask = np.ones(train_length, dtype=bool)
        mask[idx_to_del] = False
        indices = indices[mask]
        train_dataset = Subset(train_dataset, indices)
        train_length = len(train_dataset)         
        # train_dataset_ = copy.deepcopy(train_dataset)
        # train_dataset.targets = np.delete(train_dataset_.targets, idx_to_del, axis=0)
        # train_dataset.data = np.delete(train_dataset_.data, idx_to_del, axis=0)
        # del train_dataset_

    weights_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt['b'], shuffle=False, pin_memory=True) # used for the computation of the weights
    if opt['sampler'] == 'ufoym':
        weights_init = np.get_imbalance_weights(dataset,indices=indices, num_samples=None)
    else:
        weights_init = torch.DoubleTensor(np.zeros(train_length) + 0.01)

    sampler = torch.utils.data.WeightedRandomSampler(weights=weights_init, num_samples=int(len(weights_init)), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True, sampler=sampler)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=opt['b'], shuffle=True, num_workers=opt['j'], pin_memory=True, sampler=sampler)

    clean_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True, sampler=sampler)

    test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True)

    return train_loader, test_loader, clean_train_loader, weights_loader

def get_dataset_len(name):
    d = dict(cifar10=50000)
    return d[name]
