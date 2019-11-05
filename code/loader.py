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
from exptutils import get_num_classes
from PIL import Image
import os
import pickle as pkl 
import pandas as pd 
from tqdm import tqdm
#from libKMCUDA import kmeans_cuda
from utils import *
from exptutils import logical_index
from IPython import embed

data_ingredient = Ingredient('dataset')

@data_ingredient.config
def cfg():
    name = 'cifar10'  # dataset filename
    source = '../data/'
    shuffle = True #only for training set by default
    frac = 1 # fraction of dataset used
    norm = False
    perc = 0 # percentage of most exemples to be removed
    mode = 0 # remove (input weights):                         most difficult (0) | easy samples (1) | random (2)
             # remove (clustering of input weights dynamics):  farthest (3)       | nearest (4)      | random (5)  
    pilot_samp = 'default' # sampler used to train the pilot net: default | invtunnel | tunnel | ufoym 
    pilot_arch = 'allcnn' # architecture used for the pilot net
    mode_source = 'counter' # counter | cum_loss | w_mean
    num_clusters = 500 # number of clusters if mode = 3, 4, 5
    use_perc_diff = False # clustering performed on perc differences of weights
    celeba_class_attr = 'Smiling' # attribute used for binary classification in celebA

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
        elif  data == 'imagenet':
            source = source + 'imagenet/'
            ddir = source + 'train/' if train else source+'val/'
            self.data = datasets.ImageFolder(ddir, transform=transform)
        elif  data == 'cinic':
            source = source + 'CINIC/'
            ddir = source + 'train/' if train else source+'test/'
            self.data = datasets.ImageFolder(ddir, transform=transform)

            
    def __getitem__(self, index):
        data, target = self.data[index][0], self.data[index][1]
        return data, target, index

    def __len__(self):
        return len(self.data)


def corrupt_labels(train_labels, corrupt_prob, num_classes):
    labels = np.array(train_labels)
    np.random.seed(12345)
    #mask = np.random.rand(len(labels)) <= corrupt_prob
    #rnd_labels = np.random.choice(num_classes, mask.sum())
    rnd_labels = np.random.choice(num_classes, int(len(train_labels) * corrupt_prob))
    indices = np.random.choice(np.arange(len(labels)), int(len(train_labels) * corrupt_prob), replace=False)
    #labels[mask] = rnd_labels
    #labels = [int(x) for x in labels]
    for i, idx in enumerate(indices):
        while labels[idx] == rnd_labels[i]:
            rnd_labels[i] = np.random.choice(num_classes, 1)
        labels[idx] = rnd_labels[i]
    return labels, indices
    #return labels, mask


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

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, attr, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df[attr].values
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            data = self.transform(img)
        
        target = self.y[index]

        return data, target, index

class Lighting(object):
    # Lighting noise (AlexNet - style PCA - based noise)

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
IMAGENET_PCA = {
    'eigval':torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


TRAIN_TRANSFORMS_224 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        Lighting(0.05, IMAGENET_PCA['eigval'], 
                      IMAGENET_PCA['eigvec'])
    ])
TEST_TRANSFORMS_224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

@data_ingredient.capture
def load_data(name, source, shuffle, frac, perc, mode, \
        pilot_samp, pilot_arch, num_clusters, use_perc_diff, \
        celeba_class_attr, norm, mode_source, opt):

    if name == 'cifar10':
        # CIFAR_MEAN = ch.tensor([0.4914, 0.4822, 0.4465])
     #    CIFAR_STD = ch.tensor([0.2023, 0.1994, 0.2010])
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
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])

        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                               transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

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
    elif name == 'imagenet_lt' or name=='places_lt':
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

    elif name == 'celeba':
        transform_train = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.1,.1,.1),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    elif name == 'imagenet':
        transform_train = TRAIN_TRANSFORMS_224
        transform_test  = TEST_TRANSFORMS_224
    
    elif name == 'cinic':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)])

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
    elif name == 'places_lt':
        train_dataset = LT_Dataset(root='/home/matteo/data/places365_standard', 
                            txt='./data/Places_LT/Places_LT_train.txt', 
                            transform=transform_train)
        test_dataset = LT_Dataset(root='/home/matteo/data/places365_standard', 
                            txt='./data/Places_LT/Places_LT_test.txt', 
                            transform=transform_test)

    elif name in ['imagenet', 'cifar10', 'cinic','cifar100', 'mnist', 'cifar10.1', 'tinyimagenet64']:
        train_dataset = MyDataset(name, source, train=True, download=True,
                        transform=transform_train)    
        test_dataset = MyDataset(name, source, train=False, download=True,
                        transform=transform_test)
    elif name == 'celeba':
        train_dataset = CelebaDataset(csv_path='/home/aquarium/celebA/celeba_train_all_attributes.csv',
            img_dir='/home/aquarium/celebA/img_align_celeba/',
            attr = celeba_class_attr,
            transform=transform_train)
        test_dataset = CelebaDataset(csv_path='/home/aquarium/celebA/celeba_test_all_attributes.csv',
            img_dir='/home/aquarium/celebA/img_align_celeba/',
            attr = celeba_class_attr,
            transform=transform_test)
    else:
        raise NotImplementedError

    train_length = len(train_dataset)
    test_length = len(test_dataset)

    indices = np.arange(0,train_length)
    num_classes = get_num_classes(opt)
    if perc > 0:
        print('Dataset reduction of ', perc)
        if mode in [0, 1, 2]:
            # remove according to the input weights
            fn = pilot_fn = '_'.join(('pilot_', name, pilot_arch, pilot_samp, str(opt['temperature'])))
            #fn = pilot_fn = 'pilot_' + name + '_' + pilot_arch + '_' + pilot_samp
            with open(os.path.join(opt['o'], 'pilots', fn + '.pkl'), 'rb') as f:
                pilot = pkl.load(f)
            sd_idx = np.squeeze(pilot[mode_source]['sorted_idx'])
            if mode == 0:
                idx = sd_idx[int((1 - perc) * train_length):]
            elif mode == 1:
                idx = sd_idx[:int(perc * train_length)]
            elif mode == 2:
                sd_idx = np.random.permutation(sd_idx)
                idx = sd_idx[:int(perc * train_length)]
        elif mode in [3, 4, 5]:
            # remove according to the clustering of input weights dynamics
            fn = pilot_fn = 'clustering_pilot_' + name + '_' + pilot_arch + '_' + pilot_samp
            with open(os.path.join(opt['o'], 'pilots', fn + '.pkl'), 'rb') as f:
                pilot = pkl.load(f)
            weights_all_epochs = pilot['weights_all_epochs']
            if use_perc_diff:
                perc_diff = compute_perc_diff(weights_all_epochs)
                centroids, assignments = kmeans_cuda(perc_diff, num_clusters, verbosity=0, seed=opt['seed'])
                idx = get_clustering_indices_to_remove(perc_diff, centroids, assignments, perc, mode, opt['seed'])
            else:
                centroids, assignments = kmeans_cuda(weights_all_epochs, num_clusters, verbosity=0, seed=opt['seed'])
                idx = get_clustering_indices_to_remove(weights_all_epochs, centroids, assignments, perc, mode, opt['seed'])
        else:
            raise(ValueError('Valid mode values: 0,1,2'))
        mask = np.ones(train_length, dtype=bool)
        mask[idx] = False
        indices = indices[mask]
        # train_dataset = Subset(train_dataset, indices)
        train_dataset.data.data = train_dataset.data.data[indices,:,:,:]
        train_dataset.data.targets = [train_dataset.data.targets[k] for k in list(indices)]
        train_length = len(train_dataset)
        print('New Dataset length is ', train_length)


    if opt['corr_labels'] > 0:
        corr_labels, indices = corrupt_labels(train_dataset.data.targets, opt['corr_labels'], num_classes)
        train_dataset.data.targets = corr_labels
        fn = os.path.join(opt.get('o'), opt['exp'], opt['filename'])
        with open(os.path.join(fn, 'indices_corr_' + str(opt['corr_labels']) + '.pkl'), 'wb') as handle:
            pkl.dump(indices, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # if frac < 1:
    #     indices = indices[:int(frac * train_length)]
    #     train_dataset = Subset(train_dataset, indices)
    #     train_length = len(train_dataset)

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
    if opt['sampler'] == 'class':
        label_to_count = {}

        try:
            labels = train_dataset.data.targets
        except:
            labels = train_dataset.labels

        for label in labels:
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # temp = torch.utils.data.DataLoader(
        # train_dataset, batch_size=128, shuffle=False, num_workers=opt['j'], pin_memory=False)
        labels = torch.tensor(labels).cuda(opt['g'])
        weights_init = torch.zeros_like(labels).double()
        norm_ = np.array(list(label_to_count.values())).sum()

        for k,v in label_to_count.items():
            weights_init[labels == k] = norm_ / v

        #weights_init = np.get_imbalance_weights(dataset, indices=indices, num_samples=None)
    else:
        if opt['sampler'] == 'tunnel':
            sc = 1e-4
        else:
            sc = 1e4
        weights_init = torch.DoubleTensor(np.zeros(train_length) + sc)

    sampler = torch.utils.data.WeightedRandomSampler(weights=weights_init, num_samples=int(len(weights_init)), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=False, sampler=sampler)

    clean_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=min(opt['b']*5, 1024), shuffle=False, num_workers=opt['j'], pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt['b'], shuffle=False, num_workers=opt['j'], pin_memory=False)

    return train_loader, clean_train_loader, test_loader, weights_loader, train_length

def get_dataset_len(name):
    d = dict(cifar10=50000, cifar100=50000, tinyimagenet64=100000, imagenet_lt=115846, imagenet=1281167)
    return d[name]
