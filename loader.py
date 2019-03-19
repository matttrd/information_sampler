from __future__ import division, print_function, unicode_literals
from sacred import Ingredient, Experiment
from torchvision import datasets, transforms
import torch 

data_ingredient = Ingredient('dataset')

@data_ingredient.config
def cfg():
    name = 'cifar10'  # dataset filename
    source = '../data/'
    shuffle = True #only for training set by default

@data_ingredient.capture
def load_data(name, source, shuffle, opt):

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[name.upper()](source, train=True, download=True,
                       transform=transform_train),
        batch_size=opt['b'], shuffle=shuffle, num_workers=opt['j'], pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.__dict__[name.upper()](source, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True)
    return train_loader, test_loader