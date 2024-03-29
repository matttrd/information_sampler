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
import json

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

def img_num(cifar_version):
    dt = {'10': 5000, '100': 500}
    return dt[cifar_version]

def get_img_num_per_cls(cifar_version, imb_factor):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    cls_num = int(cifar_version)
    img_max = img_num(cifar_version)
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


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


class Reduced_Dataset(Dataset):
    def __init__(self, root, txt, indices, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for img_id, line in enumerate(f):
                if indices is not None:
                    if img_id in indices:
                        self.img_path.append(os.path.join(root, line.split()[0]))
                        self.labels.append(int(line.split()[1]))
                else:
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

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic

def default_loader(path):
    return Image.open(path).convert('RGB')

class INAT(Dataset):
    def __init__(self, root, ann_file, transform):

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # # print out some stats
        # print '\t' + str(len(self.imgs)) + ' images'
        # print '\t' + str(len(set(self.classes))) + ' classes'

        self.root = root
        # self.is_train = is_train
        self.loader = default_loader
        self.transform = transform

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]

        if self.transform is not None:
            img = self.transform(img)

        return img, species_id, index#, tax_ids

    def __len__(self):
        return len(self.imgs)

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
    elif name == 'imagenet_lt' or name=='places_lt' or name=='inaturalist':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.25),
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
        root = '/home/matteo/data/imagenet'
        train_dataset = LT_Dataset(root=root, 
                            txt='./data/ImageNet_LT/ImageNet_LT_train.txt', 
                            transform=transform_train)
        test_dataset = LT_Dataset(root='/home/matteo/data/imagenet', 
                            txt='./data/ImageNet_LT/ImageNet_LT_test.txt', 
                            transform=transform_test)
    elif name == 'places_lt':
        root = '/home/matteo/data/places365_standard'
        train_dataset = LT_Dataset(root=root, 
                            txt='./data/Places_LT/Places_LT_train.txt', 
                            transform=transform_train)
        test_dataset = LT_Dataset(root='/home/matteo/data/places365_standard', 
                            txt='./data/Places_LT/Places_LT_test.txt', 
                            transform=transform_test)

    elif name == 'inaturalist':
        root = '/home/matteo/data/inaturalist/'
        train_dataset = INAT(root=root, 
                             ann_file='/home/matteo/data/inaturalist/train2019.json',
                             transform=transform_train)
        test_dataset = INAT(root='/home/matteo/data/inaturalist/', 
                             ann_file='/home/matteo/data/inaturalist/val2019.json',
                             transform=transform_test)

    elif name in ['imagenet', 'cifar10','cifar100', 'mnist', 'cifar10.1', 'tinyimagenet64']:
        train_dataset = MyDataset(name, source, train=True, download=True,
                        transform=transform_train)    
        test_dataset = MyDataset(name, source, train=False, download=True,
                        transform=transform_test)
    elif name in ['cinic']:
        root = '/home/matteo/data/CINIC/'
        train_dataset = Reduced_Dataset(root=root, 
                            txt='./data/CINIC/CINIC_train.txt', indices=None,
                            transform=transform_train)
        # train_dataset = MyDataset(name, source, train=True, download=True,
        #                 transform=transform_train)
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
        if mode in [0, 1, 2, 6]:
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
            elif mode == 6:
                halfway = len(sd_idx) // 2
                idx = sd_idx[halfway - int(perc/2 * train_length):halfway + int(perc/2 * train_length)]

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

        if 'cifar' in name:
            train_dataset.data.data = train_dataset.data.data[indices,:,:,:]
            train_dataset.data.targets = [train_dataset.data.targets[k] for k in list(indices)]
            print('New Dataset length is ', train_length)
        else:
            train_dataset = Reduced_Dataset(root=root, 
                            txt='./data/CINIC/CINIC_train.txt', indices=indices,
                            transform=transform_train)
        train_length = len(train_dataset)

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

    if opt['cifar_imb_factor'] is not None and name in ['cifar10', 'cifar100']:
        if name == 'cifar10':
            img_num_per_cls = get_img_num_per_cls('10', opt['cifar_imb_factor'])
        elif name == 'cifar100':
            img_num_per_cls = get_img_num_per_cls('100', opt['cifar_imb_factor'])
        indices = []
        y = np.array(train_dataset.data.targets)
        for class_idx in range(len(img_num_per_cls)):
            # position of samples belonging to class_idx
            current_class_all_idx = np.argwhere(y == class_idx)
            # convert the result into a 1-D list
            current_class_all_idx = list(current_class_all_idx[:,0])
            current_class_selected_idx = list(np.random.choice(current_class_all_idx, img_num_per_cls[class_idx], replace=False))
            indices = indices + current_class_selected_idx
        train_dataset.data.data = train_dataset.data.data[indices,:,:,:]
        train_dataset.data.targets = [train_dataset.data.targets[k] for k in list(indices)]
        train_length = len(train_dataset)
        print('New Dataset length is ', train_length)
        #print(sum(img_num_per_cls))   # sanity check: must be equal to 'train_length'
        #print(img_num_per_cls) 
        fn = os.path.join(opt.get('o'), opt['exp'], opt['filename'])
        with open(os.path.join(fn, 'selected_indices.pkl'), 'wb') as handle:
            pkl.dump(indices, handle, protocol=pkl.HIGHEST_PROTOCOL)
    

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
            sc = 1e-8
        else:
            sc = 1e8
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
