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
from PIL import Image
import os
import pickle as pkl 

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
    pilot_samp = 'default' # sampler used to train the pilot net: default | invtunnel | tunnel | ufoym 
    pilot_arch = 'allcnn' # architecture used for the pilot net
    pilot_ep = 1 # epochs in the training of the pilot net

class MyDataset(Dataset):
    def __init__(self, data, source, train, download, transform):
        if data == 'cifar10':
            self.data =  datasets.CIFAR10(source, train=train, download=download, transform=transform)
        elif data == 'cifar100':
            self.data =  datasets.CIFAR100(source, train=train, download=download, transform=transform)
        elif data == 'tinyimagenet64':
            source = source + 'tiny-imagenet-200/'
            ddir = source + 'train/' if train else source+'val/'
            self.data = datasets.ImageFolder(ddir, transform=transform)
        else:
            print('Only CIFAR 10/100 allowed!')

    def __getitem__(self, index):
        data, target = self.data[index][0], self.data[index][1]
        return data, target, index

    def __len__(self):
        return len(self.data)

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            data = self.transform(sample)

        return data, target, index


@data_ingredient.capture
def load_data(name, source, shuffle, frac, perc, mode, pilot_samp, pilot_arch, pilot_ep, norm, opt):

    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))])

        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))])
    
    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))])

        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))])

    elif name == 'mnist':
        transform_train = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                           ])
        transform_test = transforms.Compose([
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

    elif name == 'tinyimagenet64':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if norm else
                 transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])
    elif name == 'imagenet_lt':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
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

    if name == 'imagenet_lt':
        train_dataset = LT_Dataset(root='/home/matteo/data/imagenet', 
                            txt='./data/ImageNet_LT/ImageNet_LT_train.txt', 
                            transform=transform_train)
        test_dataset = LT_Dataset(root='/home/matteo/data/imagenet', 
                            txt='./data/ImageNet_LT/ImageNet_LT_test.txt', 
                            transform=transform_test)
    elif name in ['cifar10', 'cifar100', 'mnist', 'cifar10.1', 'tinyimagenet64']:
        train_dataset = MyDataset(name, source, train=True, download=True,
                        transform=transform_train)    
        test_dataset = MyDataset(name, source, train=False, download=True,
                        transform=transform_test)
    
    else:
        raise NotImplementedError

    train_length = len(train_dataset)
    test_length = len(test_dataset)

    indices = np.arange(0,train_length)
    num_classes = get_num_classes(opt)
    if perc > 0:
        print('Dataset reduction of ', perc)
        fn = pilot_fn = 'pilot_' + name + '_' + pilot_arch + '_' + pilot_samp + '_' + str(pilot_ep) + '_epochs'
        with open(os.path.join(opt['o'], opt['exp'], 'pilots', fn + '.pkl'), 'rb') as f:
            pilot = pkl.load(f)
        sd_idx = np.squeeze(pilot['sorted_idx'])
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

        train_length = len(train_dataset)
        print('New Dataset length is ', train_length)

    if frac < 1:
        indices = indices[:int(frac * train_length)]
        train_dataset = Subset(train_dataset, indices)
        train_length = len(train_dataset)

    if opt['classes'] is not None:
        sclasses = opt['classes']
        num_classes = len(opt['classes'])
        
        def reduce_dataset(dataset, classes):
            # remove all classes but the two
            if isinstance(dataset, Subset):
                dataset = dataset.dataset
            
            labels = dataset.data.targets
            indices = np.arange(0,len(dataset.data))

            selector = list(map(lambda x: x in classes, labels))
            indices = indices[selector]
            dataset = Subset(dataset, indices)
            return dataset, indices
        
        train_dataset, ind = reduce_dataset(train_dataset, sclasses)
        test_dataset, _ = reduce_dataset(test_dataset, sclasses)
        train_length = len(train_dataset)

    

    if opt['unbalanced']: 
        class_labels = range(num_classes)
        sample_probs = torch.rand(num_classes)

        if opt['classes']:
            mapper = dict(zip(opt['classes'], list(range(num_classes))))
        else:
            mapper = dict(zip(list(range(num_classes)), list(range(num_classes))))
        # if isinstance(train_dataset, Subset):
        #     train_dataset = train_dataset.dataset

        idx_to_del = [i for i, (_, label, idx) in enumerate(train_dataset)
                      if 0.7*random.random() > sample_probs[mapper[label]]]

        mask = np.ones(train_length, dtype=bool)
        indices = np.arange(0,train_length)
        mask[idx_to_del] = False
        indices = indices[mask]
        train_dataset = Subset(train_dataset, indices)
        train_length = len(train_dataset)   

    weights_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True) # used for the computation of the weights
    if opt['sampler'] == 'ufoym':
        weights_init = np.get_imbalance_weights(dataset, indices=indices, num_samples=None)
    else:
        weights_init = torch.DoubleTensor(np.zeros(train_length) + 0.01)

    sampler = torch.utils.data.WeightedRandomSampler(weights=weights_init, num_samples=int(len(weights_init)), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True, sampler=sampler)

    test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True)

    return train_loader, test_loader, weights_loader

def get_dataset_len(name):
    d = dict(cifar10=50000, cifar100=50000, tinyimagenet64=100000, imagenet_lt=115846)
    return d[name]