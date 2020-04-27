import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.torch import ToTensor

from transforms import test_transform, mix_transform, mix_transform2
from youtrain.factory import DataFactory
from samplers import WeightedSampler
from utils import onehot


class BaseDataset(Dataset):
    def __init__(self, ids, transform):
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        raise NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self, ids, transform, num_classes):
        super().__init__(ids, transform)
        self.num_classes = num_classes
        self.mapping = {'2_long': 0, '3_medium': 1, '4_closeup': 2, '5_detail': 3}
        self.sampler = WeightedSampler(self)

        self.mixup = True
        self.mixup_p = 0.1
        self.alpha = 0.5

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()

        line_1 = self.ids.iloc[index]
        label_1 = np.array([self.mapping[line_1['label']]])
        image_1 = cv2.imread(line_1['path'])

        if self.mixup and np.random.uniform(0, 1) > self.mixup_p:
            while True:
                idx = next(iter(self.sampler)).item()    # generate idx with self.sampler
                line_2 = self.ids.iloc[idx]
                label_2 = np.array([self.mapping[line_2['label']]])
                if label_1 != label_2:
                    break

            image_2 = cv2.imread(line_2['path'])
            image_1 = self.transform(image=image_1)['image']
            image_2 = self.transform(image=image_2)['image']

            label_1 = ToTensor()(image=label_1)['image']
            label_2 = ToTensor()(image=label_2)['image']
            label_1 = onehot(label_1, self.num_classes)
            label_2 = onehot(label_2, self.num_classes)

            _lambda = np.random.beta(self.alpha, self.alpha)
            images = _lambda * image_1 + (1 - _lambda) * image_2
            labels = _lambda * label_1 + (1 - _lambda) * label_2

        else:
            images = self.transform(image=image_1)['image']
            label_1 = ToTensor()(image=label_1)['image']
            labels = onehot(label_1, self.num_classes)

        return {'image': images, 'mask': labels}


class ValDataset(BaseDataset):
    def __init__(self, ids, transform, num_classes):
        self.ids = pd.read_csv('/mnt/hdd2/datasets/naive_data/shot_dataset/shot_total_bigger/folds_val.csv')
        super().__init__(self.ids, transform)
        self.num_classes = num_classes
        self.mapping = {'2_long': 0, '3_medium': 1, '4_closeup': 2, '5_detail': 3}

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        line = self.ids.iloc[index]

        image = cv2.imread(line['path'])
        image = self.transform(image=image)['image']

        label = np.array([self.mapping[line['label']]])
        label = ToTensor()(image=label)['image']
        label = onehot(label, self.num_classes)

        return {'image': image, 'mask': label}


class BadVideoValDataset(BaseDataset):
    def __init__(self, ids, transform, num_classes):
        print('BadVideoValDataset')
        super().__init__(ids, transform)
        self.num_classes = num_classes
        self.mapping = {'2_long': 0, '3_medium': 1, '4_closeup': 2, '5_detail': 3}
        # self.ids = pd.read_csv('/mnt/hdd2/datasets/naive_data/shot_dataset/shot_total_bigger/bad_video_folds_val.csv')

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        line = self.ids.iloc[index]

        image = cv2.imread(line['path'])
        image = self.transform(image=image)['image']

        label = np.array([self.mapping[line['label']]])
        label = ToTensor()(image=label)['image']
        label = onehot(label, self.num_classes)

        return {'image': image, 'mask': label}


class TestDataset(BaseDataset):
    def __init__(self, image_dir, transform):
        print('TestDataset')
        self.ids = glob.glob(os.path.join(image_dir, '*.*'))
        super().__init__(self.ids, transform)
        self.mapping = {'2_long': 0, '3_medium': 1, '4_closeup': 2, '5_detail': 3}

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        path = self.ids[index]
        image = cv2.imread(path)
        image = self.transform(image=image)['image']
        print(image)
        print(path)
        return torch.stack([image])


class TaskDataFactory(DataFactory):
    def __init__(self, params, paths, **kwargs):
        super().__init__(params, paths, **kwargs)
        self.fold = kwargs['fold']
        self._folds = None
        self.num_classes = kwargs['num_classes']

    @property
    def data_path(self):
        return Path(self.paths['path'])

    def make_transform(self, stage, is_train=False):
        if is_train:
            if stage['augmentation'] == 'mix_transform':
                transform = mix_transform(**self.params['augmentation_params'])
            elif stage['augmentation'] == 'mix_transform2':
                transform = mix_transform2(**self.params['augmentation_params'])
            else:
                raise KeyError('augmentation does not found')
        else:
            transform = test_transform(**self.params['augmentation_params'])
        return transform

    def make_dataset(self, stage, is_train):
        transform = self.make_transform(stage, is_train)
        ids = self.train_ids if is_train else self.val_ids
        if is_train:
            return TrainDataset(ids=ids, transform=transform, num_classes=self.num_classes)
        else:
            print('make_dataset no train')
            return ValDataset(ids=ids, transform=transform, num_classes=self.num_classes)
            # return BadVideoValDataset(ids=ids, transform=transform, num_classes=self.num_classes)

    def make_loader(self, stage, is_train=False):
        dataset = self.make_dataset(stage, is_train)
        sampler = WeightedSampler(dataset) if is_train else None
        return DataLoader(
            dataset=dataset,
            batch_size=self.params['batch_size'],
            shuffle=not bool(sampler),
            drop_last=is_train,
            num_workers=self.params['num_workers'],
            pin_memory=torch.cuda.is_available(),
            sampler=sampler
        )

    @property
    def folds(self):
        if self._folds is None:
            print(self.data_path,self.paths['folds'])
            self._folds = pd.read_csv(self.data_path / self.paths['folds'])
        return self._folds

    @property
    def train_ids(self):
        return self.folds#.loc[self.folds['fold'] != self.fold]

    @property
    def val_ids(self):
        return self.folds.loc[self.folds['fold'] == self.fold]
