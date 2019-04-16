from __future__ import division, print_function, unicode_literals
from sacred import Ingredient, Experiment
from torchvision import datasets, transforms
import torchnet as tnt
import torch 
import numpy as np
import copy
import random

data_ingredient = Ingredient('dataset')

@data_ingredient.config
def cfg():
    name = 'cifar10'  # dataset filename
    source = '../data/'
    shuffle = True #only for training set by default
    frac = 1 # fraction of dataset used
    norm = False


@data_ingredient.capture
def load_data(name, source, shuffle, frac, norm, opt):

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
    
    train_dataset = datasets.__dict__[name.upper()](source, train=True, download=True,
                       transform=transform_train)
    clean_train_dataset = datasets.__dict__[name.upper()](source, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))]))
    
    test_dataset = datasets.__dict__[name.upper()](source, train=False, download=True,
                       transform=transforms_test)

    train_length = len(train_dataset)
    test_length = len(test_dataset)

    if frac < 1:
        train_dataset.targets = train_dataset.targets[:int(frac * train_length)]
        train_dataset.data = train_dataset.data[:int(frac * train_length)]
        test_dataset.targets = test_dataset.targets[:int(frac * test_length)]
        test_dataset.data = test_dataset.data[:int(frac * test_length)]
        train_length = len(train_dataset)
        test_length = len(test_dataset)


    dataset_length = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['b'], shuffle=shuffle, num_workers=opt['j'], pin_memory=True)

    clean_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True)

    return train_loader, test_loader, clean_train_loader
