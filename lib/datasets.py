from attrdict import AttrDict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.cifar import VisionDataset

from lib.utils import ROOT


class MyVisionDataset(VisionDataset):
    def __init__(self, images, labels, transform, labels_sparse=None):
        super().__init__(ROOT+'/data', transform=transform)
        self.data, self.targets = images, labels
        self.targets = torch.asarray(self.targets, dtype=torch.int64)
        self.targets_sparse = labels_sparse
        if self.targets_sparse is not None:
            self.targets_sparse = torch.asarray(self.targets_sparse, dtype=torch.int64)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # img = img.numpy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.targets_sparse is not None:
            target_sparse = int(self.targets_sparse[index])
            return img, target, target_sparse
        else:
            return img, target

    def __len__(self):
        return len(self.data)


# TODO: will have to extend with sparse_labels
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

        #### Single dataloader version
        # dataset = MyVisionDataset(images, labels, transform)
        # self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=cntx_pts_per_class*n_classes, shuffle=True, drop_last=True, **kwargs)
        # self.data_iter = iter(self.dataloader)

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

        #### Single dataloader version
        # try:
        #     input_all, target_all = next(self.data_iter)
        # except StopIteration:
        #     self.data_iter = iter(self.dataloader)
        #     input_all, target_all = next(self.data_iter)
        # input_all, target_all = input_all.to(self.device), target_all.to(self.device)

        return input_all, target_all

    def sample(self, n_experts=1):
        # input_lst = []
        # target_lst = []
        # for _ in range(n_experts):
        #     input, target = self._balanced_sample()
        #     input_lst.append(input.unsqueeze(0))
        #     target_lst.append(target.unsqueeze(0))
        # cntx = AttrDict()
        # cntx.xc = torch.vstack(input_lst)
        # cntx.yc = torch.vstack(target_lst)
        
        ## Since only using {yc,mc} not necessary to resample for multiple experts
        input, target = self._balanced_sample()
        cntx = AttrDict()
        cntx.xc = input.unsqueeze(0).repeat(n_experts,1,1,1,1)
        cntx.yc = target.unsqueeze(0).repeat(n_experts,1)

        return cntx
    
    def reset(self):
        for cc in range(self.n_classes):
            self.data_iter_lst[cc] = iter(self.dataloader_lst[cc])


# From https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def load_cifar(variety='10', data_aug=False, seed=0):
    assert variety in ['10','20_100']
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

    if variety == '10':
        train_dataset_all = datasets.CIFAR10(root=ROOT+'/data', train=True, download=True, transform=transform_train)
        train_dataset_all_without_da = datasets.CIFAR10(root=ROOT+'/data', train=True, download=True, transform=transform_test)
        test_dataset = datasets.CIFAR10(root=ROOT+'/data', train=False, download=True, transform=transform_test)
    else:
        train_dataset_all = datasets.CIFAR100(root=ROOT+'/data', train=True, download=True, transform=transform_train)
        targets_sparse = train_dataset_all.targets # {0,99}
        targets_coarse = sparse2coarse(targets_sparse).tolist() # {0,19}
        train_dataset_all = MyVisionDataset(train_dataset_all.data, targets_coarse, transform_train, targets_sparse)
        train_dataset_all_without_da = MyVisionDataset(train_dataset_all.data, targets_coarse, transform_test, targets_sparse)

        test_dataset = datasets.CIFAR100(root=ROOT+'/data', train=False, download=True, transform=transform_test)
        test_targets_sparse = test_dataset.targets # {0,99}
        test_targets_coarse = sparse2coarse(test_targets_sparse).tolist() # {0,19}
        test_dataset = MyVisionDataset(test_dataset.data, test_targets_coarse, transform_test, test_targets_sparse)

    # 90/10 split for train/val size
    train_size = int(0.90 * len(train_dataset_all))
    val_size = len(train_dataset_all) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset_all, [train_size, val_size], \
                                                     generator=torch.Generator().manual_seed(seed))
    
    # Need valid dataset without data augmentation
    _, val_dataset = torch.utils.data.random_split(train_dataset_all_without_da, [train_size, val_size], \
                                                   generator=torch.Generator().manual_seed(seed))

    return train_dataset, val_dataset, test_dataset
