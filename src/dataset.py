import cv2
import os
import glob
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from youtrain.factory import DataFactory
from transforms import test_transform, mix_transform, mix_transform2
from albumentations.torch import ToTensor
from samplers import WeightedSampler


class BaseDataset(Dataset):
    def __init__(self, ids, transform):
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        raise NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self, ids, transform):
        super().__init__(ids, transform)
        self.mapping = {'2_long': 0, '3_medium': 1, '4_closeup': 2, '5_detail': 3}

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        line = self.ids.iloc[index]
        image = cv2.imread(line['path'])
        result = self.transform(image=image)
        mask = np.array([self.mapping[line['label']]])
        result['mask'] = ToTensor()(image=mask)['image']
        return result


class TestDataset(BaseDataset):
    def __init__(self, image_dir, transform):
        ids = glob.glob(os.path.join(image_dir, '**/*.*'), recursive=True)
        super().__init__(ids, transform)
        self.transform = transform
        self.ids = ids
        self.image_dir = image_dir

    def __getitem__(self, index):
        name = self.ids[index]
        image = cv2.imread(name)
        return self.transform(image=image)['image']


class TaskDataFactory(DataFactory):
    def __init__(self, params, paths, **kwargs):
        super().__init__(params, paths, **kwargs)
        self.fold = kwargs['fold']
        self._folds = None
        # self.class_weights = {0: 1065, 1: 1864, 2: 217, 3: 461}
        # self.class_weights = {0: 26701, 1: 37250, 2: 5752, 3: 1639}
        self.class_weights = {0: 814, 1: 3072, 2: 1084, 3: 1354}

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
        return TrainDataset(
            ids=ids,
            transform=transform)

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
            self._folds = pd.read_csv(self.data_path / self.paths['folds'])
        return self._folds

    @property
    def train_ids(self):
        return self.folds.loc[self.folds['fold'] != self.fold]

    @property
    def val_ids(self):
        return self.folds.loc[self.folds['fold'] == self.fold]
