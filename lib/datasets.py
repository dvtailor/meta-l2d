from attrdict import AttrDict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.cifar import VisionDataset

from lib.utils import ROOT


class MyVisionDataset(VisionDataset):
    def __init__(self, images, labels, transform):
        super().__init__(ROOT+'/data', transform=transform)
        self.data, self.targets = images, labels
        self.targets = torch.asarray(self.targets, dtype=torch.int64)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # img = img.numpy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


# May want to extend to allow variable context set sizes
#  i.e. min_cntx_pts_per_class, max_cntx_pts_per_class
# Since batch size in data loader is fixed, just specify max value and then take subset
class ContextSampler():
    def __init__(self, images, labels, transform, cntx_pts_per_class=5, n_classes=10, device='cpu', **kwargs):
        self.cntx_pts_per_class = cntx_pts_per_class
        self.n_classes = n_classes
        self.device = device

        self.dataloader_lst = [] # separated by class
        self.data_iter_lst = []
        for cc in range(n_classes):
            indices = np.where(labels==cc)[0]
            dataset = MyVisionDataset(images[indices], labels[indices], transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=cntx_pts_per_class, shuffle=True, drop_last=True, **kwargs)
            self.dataloader_lst.append(dataloader)
            self.data_iter_lst.append(iter(dataloader))

    # Sample context points in a strictly class balanced way
    def _balanced_sample(self):
        input_lst = []
        target_lst = []
        for cc in range(self.n_classes):
            try:
                input, target = next(self.data_iter_lst[cc])
            except StopIteration:
                self.data_iter_lst[cc] = iter(self.dataloader_lst[cc])
                input, target = next(self.data_iter_lst[cc])
            input_lst.append(input)
            target_lst.append(target)
        perm = torch.randperm(self.cntx_pts_per_class*self.n_classes)
        input_all = torch.vstack(input_lst)[perm]
        target_all = torch.cat(target_lst)[perm]
        input_all, target_all = input_all.to(self.device), target_all.to(self.device)
        return input_all, target_all

    def sample(self, n_experts=1):
        input_lst = []
        target_lst = []
        for _ in range(n_experts):
            input, target = self._balanced_sample()
            input_lst.append(input.unsqueeze(0))
            target_lst.append(target.unsqueeze(0))
        cntx = AttrDict()
        cntx.xc = torch.vstack(input_lst)
        cntx.yc = torch.vstack(target_lst)
        return cntx
    
    def reset(self):
        for cc in range(self.n_classes):
            self.data_iter_lst[cc] = iter(self.dataloader_lst[cc])


def load_cifar10(data_aug=False, seed=0):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                            (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # 90/10 split for train/val size
    train_dataset_all = datasets.CIFAR10(root=ROOT+'/data', train=True, download=True, transform=transform_train)
    train_size = int(0.90 * len(train_dataset_all))
    val_size = len(train_dataset_all) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset_all, [train_size, val_size], \
                                                     generator=torch.Generator().manual_seed(seed))
    # Need valid dataset without data augmentation
    train_dataset_all = datasets.CIFAR10(root=ROOT+'/data', train=True, download=True, transform=transform_test)
    train_size = int(0.90 * len(train_dataset_all))
    val_size = len(train_dataset_all) - train_size
    _, val_dataset = torch.utils.data.random_split(train_dataset_all, [train_size, val_size], \
                                                   generator=torch.Generator().manual_seed(seed))
    
    test_dataset = datasets.CIFAR10(root=ROOT+'/data', train=False, download=True, transform=transform_test)

    return train_dataset, val_dataset, test_dataset


class UnNormalize(transforms.Normalize):
    def __init__(self,mean,std,*args,**kwargs):
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)
