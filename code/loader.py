from __future__ import division, print_function, unicode_literals
from sacred import Ingredient, Experiment
from torchvision import datasets, transforms
import torchnet as tnt
import torch 
from sampler import ImbalancedDatasetSampler
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
    perc = 0 # percentage of most difficult exemples to be removed


@data_ingredient.capture
def load_data(name, source, shuffle, frac, perc, norm, opt):

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

    if perc > 0:
        sd_idx = torch.load('sorted_datasets.pz')[name]
        idx = sd_idx[:-int(frac * train_length)]
        mask = np.ones(train_length, dtype=bool)
        mask[idx] = False
        train_dataset.train_labels = np.array(train_dataset.train_labels)[mask]
        train_dataset.train_data = np.array(train_dataset.train_data)[mask]
        train_length -= len(idx)

    if frac < 1:
        train_dataset.train_labels = train_dataset.train_labels[:int(frac * train_length)]
        train_dataset.train_data = train_dataset.train_data[:int(frac * train_length)]
        test_dataset.test_labels = test_dataset.test_labels[:int(frac * test_length)]
        test_dataset.test_data = test_dataset.test_data[:int(frac * test_length)]
        train_length = len(train_dataset)
        test_length = len(test_dataset)

    if opt['unbalanced']: 
        try:
            num_classes = len(train_dataset.train_labels.unique())
        except Exception:
            num_classes = len(np.unique(train_dataset.train_labels))

        class_labels = range(num_classes)
        sample_probs = torch.rand(num_classes)
        idx_to_del = [i for i, label in enumerate(train_dataset.train_labels) 
                      if random.random() > sample_probs[label]]
        train_dataset_ = copy.deepcopy(train_dataset)
        train_dataset.train_labels = np.delete(train_dataset_.train_labels, idx_to_del, axis=0)
        train_dataset.train_data = np.delete(train_dataset_.train_data, idx_to_del, axis=0)
        del train_dataset_


    # our sampler
    dataset_length = len(train_dataset)
    weights_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, pin_memory=True) # used for the computation of the weights

    if opt['sampler'] == 'our':
        shuffle = False
        weights_init = torch.FloatTensor(np.random.uniform(0, 1, dataset_length))
        weights_init = weights_init.cuda(opt['g'])
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights_init, num_samples=int(len(weights_init)), replacement=True)
    elif opt['sampler'] == 'ufoym':
        shuffle = False
        sampler = ImbalancedDatasetSampler(train_dataset)
        #weights_loader = None
    elif opt['sampler'] == 'default':
        sampler = None
        #weights_loader = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['b'], shuffle=shuffle, num_workers=opt['j'], pin_memory=True, sampler=sampler)

    clean_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True, sampler=sampler)

    test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True)

    return train_loader, test_loader, clean_train_loader, weights_loader

