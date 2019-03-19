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

    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transforms_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif name == 'mnist':
        transform_train = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                           ])
        transforms_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    else:
        raise NotImplementedError
    
    train_dataset = datasets.__dict__[name.upper()](source, train=True, download=True,
                       transform=transform_train)
    
    if opt['unbalanced']: 
        try:
            num_classes = len(train_dataset.train_labels.unique())
        except Exception:
            num_classes = len(np.unique(train_dataset.train_labels))

        class_labels = range(num_classes)
        sample_probs = torch.rand(num_classes)
        idx_to_del = [i for i, label in enumerate(train_dataset.dataset.train_labels) 
                      if random.random() > sample_probs[label]]
        train_dataset_ = copy.deepcopy(train_dataset)
        train_dataset.train_labels = np.delete(train_dataset_.dataset.train_labels, idx_to_del, axis=0)
        train_dataset.train_data = np.delete(train_dataset_.dataset.train_data, idx_to_del, axis=0)
        del train_dataset_

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['b'], shuffle=shuffle, num_workers=opt['j'], pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.__dict__[name.upper()](source, train=False, download=True,
                       transform=transforms_test)
        batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=True)
    return train_loader, test_loader