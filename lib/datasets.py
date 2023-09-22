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
