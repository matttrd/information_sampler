from __future__ import division, print_function, unicode_literals
from sacred import Ingredient, Experiment
from torchvision import datasets, transforms
import torch 
from sampler import ImbalancedDatasetSampler

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


    # our sampler
    dataset_length = len(train_dataset)
   
    if opt['sampler'] == 'our':
        shuffle = False
        weights_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=False, **kwargs) # used for the computation of the weights
        weights_init = torch.FloatTensor(np.random.uniform(0, 1, dataset_length))
        weights_init = weights_init.to(device)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights_init, num_samples=int(len(weights_init)), replacement=True)
    elif opt['sampler'] == 'ufoym':
        shuffle = False
        sampler = ImbalancedDatasetSampler(train_dataset)
        weights_loader = None
    elif opt['sampler'] == 'default':
        sampler = None
        weights_loader = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['b'], shuffle=shuffle, num_workers=opt['j'], pin_memory=True, sampler=sampler)

    test_loader = torch.utils.data.DataLoader(
        datasets.__dict__[name.upper()](source, train=False, download=True,
                       transform=transforms_test), 
            batch_size=opt['b'], shuffle=shuffle, num_workers=opt['j'], pin_memory=True, sampler=sampler)

    return train_loader, test_loader, weights_loader

