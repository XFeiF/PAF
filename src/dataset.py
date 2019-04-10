'''
@author: Aeolus
@url: x-fei.me
@time: 2019-04-08 18:43
'''
import os
from os.path import join, exists
import shutil
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder


class BaseLoader:
    def __init__(self, args):
        # special params
        self.num_workers = args['num_workers']
        if 'batch_size' in args:
            self.batch_size = args['batch_size']

        # custom properties
        self._dataset_train = None
        self._dataset_eval = None
        self._dataset_test = None
        self._dataset_norm = None
        self._dataset_attack = None
        self.dataloader_train = None
        self.dataloader_eval = None
        self.dataloader_test = None

    @property
    def dataset_train(self):
        raise NotImplementedError

    @property
    def dataset_eval(self):
        raise NotImplementedError

    @property
    def dataset_test(self):
        raise NotImplementedError

    @property
    def dataset_norm(self):
        raise NotImplementedError

    @property
    def train(self):
        if self.dataloader_train is None:
            self.dataloader_train = Data.DataLoader(self.dataset_train,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers,
                                                    pin_memory=True)
        return self.dataloader_train

    @property
    def eval(self):
        if self.dataloader_eval is None:
            self.dataloader_eval = Data.DataLoader(self.dataset_eval,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True)
        return self.dataloader_eval

    @property
    def test(self):
        if self.dataloader_test is None:
            self.dataloader_test = Data.DataLoader(self.dataset_test,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True)
        return self.dataloader_test

    def cal_norm(self):
        ds = self.dataset_norm
        dl = Data.DataLoader(dataset=ds, batch_size=len(ds),
                             num_workers=self.num_workers, pin_memory=True)
        for x, y in dl:
            x = x.cuda(non_blocking=True)
            print('mean: ', x[:, 0, :, :].mean().item(),
                  x[:, 1, :, :].mean().item(), x[:, 2, :, :].mean().item())
            print('std: ', x[:, 0, :, :].std().item(),
                  x[:, 1, :, :].std().item(), x[:, 2, :, :].std().item())

    @staticmethod
    def random_sample_base(base_dir, transform, size):
        classes = os.listdir(base_dir)
        classes.sort()
        images = []
        for c in classes:
            folder = join(bdir, c)
            for file_name in np.random.choice(os.listdir(folder), size, False):
                img = join(c_folder, file_name)
                images.append(transform(Image.open(img)))
        return classes, torch.stack(images)


class CIFAR10(BaseLoader):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    num_classes = 10
    class_names = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    def __init__(self, args):
        super(CIFAR10, self).__init__(args)
        self.base_dir = args['CIFAR10_dir']
        self.mean = CIFAR10.mean
        self.std = CIFAR10.std
        self.class_names = CIFAR10.class_names

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])
            self._dataset_train = torchvision.datasets.CIFAR10(
                root=self.base_dir, train=True, transform=tf)
        return self._dataset_train

    @property
    def dataset_eval(self):
        if self._dataset_eval is None:
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])
            self._dataset_eval = torchvision.datasets.CIFAR10(
                root=self.base_dir, train=False, transform=tf)
        return self._dataset_eval

    @property
    def dataset_norm(self):
        if self._dataset_norm is None:
            self._dataset_norm = torchvision.datasets.CIFAR10(
                self.base_dir, transform=transforms.ToTensor())
        return self._dataset_norm


class ImageNet100(BaseLoader):
    mean = [0.47881872, 0.45927624, 0.41515172]
    std = [0.27191086, 0.26549916, 0.27758414]
    num_classes = 100
    class_names = [str(i) for i in range(num_classes)]

    def __init__(self, args):
        super(ImageNet100, self).__init__(args)
        # base image dir
        self.base_dir = args['ImageNet100_dir']
        self.mean = ImageNet100.mean
        self.std = ImageNet100.std

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                transforms.RandomRotation([-180, 180]),
                transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                        scale=[0.7, 1.3]),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_train = torchvision.datasets.ImageFolder(
                join(self.base_dir, 'train'), transform=tf)
        return self._dataset_train

    @property
    def dataset_eval(self):
        if self._dataset_eval is None:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_eval = torchvision.datasets.ImageFolder(
                join(self.base_dir, 'val'), transform=tf)
        return self._dataset_eval

    @property
    def dataset_norm(self):
        if self._dataset_norm is None:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
            self._dataset_norm = ImageFolder(join(self.base_dir, 'train'),
                                             transform=tf)
        return self._dataset_norm

    def cal_norm(self, n=10):
        ds = self.dataset_norm
        dl = Data.DataLoader(dataset=ds, batch_size=self.batch_size,
                             num_workers=self.num_workers, pin_memory=True)
        m1 = m2 = m3 = s1 = s2 = s3 = 0
        for i in range(n):
            print('times:{}'.format(i))
            ll = len(dl)
            for idx, (x, y) in enumerate(dl):
                print('iter {} of {}'.format(idx, ll))
                x = x.cuda(non_blocking=True)
                m1 += x[:, 0, :, :].mean().item()
                m2 += x[:, 1, :, :].mean().item()
                m3 += x[:, 2, :, :].mean().item()
                s1 += x[:, 0, :, :].std().item()
                s2 += x[:, 1, :, :].std().item()
                s3 += x[:, 2, :, :].std().item()

        n = n * len(dl)
        print('mean: ', m1 / n, m2 / n, m3 / n)
        print('std: ', s1 / n, s2 / n, s3 / n)

    def random_sample(self, size):
        base_dir = join(self.base_dir, 'eval')
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor()
        ])
        return self.random_sample_base(base_dir, transform, size)


def get_dataloader(args):
    if args['dataset'] == 'ImageNet100':
        return ImageNet100(args)
    if args['dataset'] == 'CIFAR10':
        return CIFAR10(args)
    else:
        raise ValueError('No dataset: {}'.format(args['dataset']))


if __name__ == '__main__':
    args = {'dataset': 'ImageNet100',
            'ImageNet100_dir': '/data/DataSets/MyImagenet',
            'num_workers': 4, 'batch_size': 256}
    ds = get_dataloader(args)
    print('start cal')
    ds.cal_norm(n=3)
