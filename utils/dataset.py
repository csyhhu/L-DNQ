"""
An utils code for loading dataset
"""
import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from datetime import datetime
import utils.imagenet_utils as imagenet_utils
import utils.CIFAR10_utils as CIFAR10_utils


def get_mean_and_std(dataset, n_channels=3):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(n_channels):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_dataloader(dataset_name, split, batch_size, \
                   shuffle = True, drop_last = True, ratio=-1, resample=False):

    print ('[%s] Loading %s from %s' %(datetime.now(), split, dataset_name))

    loader = None

    if dataset_name == 'CIFAR10':
        data_root_list = ['/home/shangyu/CIFAR10', '/home/sinno/csy/CIFAR10', '/data/CIFAR10', '~/Documents/datasets/CIFAR10']
        # Default data root
        data_root = data_root_list[0]
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split in ['train', 'limited']:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = CIFAR10_utils.CIFAR10(root=data_root, split=split, download=True,
                                             transform=transform_train, ratio=ratio, resample=resample)
            print ('Number of training instances used: %d' %(len(trainset)))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 shuffle=shuffle, num_workers=2, drop_last=drop_last)

        elif split == 'test' or split == 'val':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                                   transform=transform_test)
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=2, drop_last=drop_last)

    elif dataset_name == 'ImageNet':
        data_root_list = ['/remote-imagenet', '/data/imagenet', '/mnt/public/imagenet']
        # Default data root
        data_root = data_root_list[0]
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        traindir = ('./train_imagenet_list.pkl', './limited_train_imagenet_list.pkl', './classes.pkl', './classes-to-idx.pkl','%s/train' %data_root)
        valdir = ('./val_imagenet_list.pkl', ' ', './classes.pkl', './classes-to-idx.pkl', '%s/val-pytorch' %data_root)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if split in ['train', 'limited']:
            trainDataset = imagenet_utils.ImageFolder(traindir, split, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), ratio=ratio, resample=resample)
            print ('Number of training data used: %d' %(len(trainDataset)))
            loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers=4 * torch.cuda.device_count(), pin_memory=True)

        elif split == 'val' or split == 'test':
            valDataset = imagenet_utils.ImageFolder(valdir, split, transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            loader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers=4 * torch.cuda.device_count(), pin_memory=True)

    assert (loader is not None)
    print ('[DATA LOADING] Loading from %s-%s finish. Number of images: %d, Number of batches: %d' \
           %(dataset_name, split, len(loader.dataset), len(loader)))

    return loader


if __name__ == '__main__':

    loader = get_dataloader('ImageNet', 'train', 128)

    for inputs, targets in loader:
        print(targets)
        input()