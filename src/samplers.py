import numpy as np
from torch.utils.data import Sampler
import torch
import pickle
import os


class VideoSampler(Sampler):
    def __init__(self, dataset, p=5):
        self.dataset = dataset
        self.p = p

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        p = 1/(1+self.p)
        self.labels = np.array(self.dataset.labels)
        self.zeros = np.where(self.labels == 0)[0]
        self.ones = np.where(self.labels == 1)[0]
        idxes = []
        ones_size = int(p*self.__len__())
        zeros_size = int((1-p) * self.__len__())
        idxes += list(np.random.choice(self.ones, ones_size, replace=True))
        idxes += list(np.random.choice(self.zeros, zeros_size, replace=True))
        np.random.shuffle(idxes)

        return iter(idxes)


class WeightedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.ids = self.dataset.ids
        label_counts = self.ids['label'].value_counts().to_dict()
        label_counts = {k: 1 / v for k, v in label_counts.items()}
        weights = self.ids['label'].map(label_counts).values
        self.weights = torch.DoubleTensor(weights)
        print('WeightedRandomSampler upload!')

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.__len__(), True))

    def __len__(self):
        return len(self.dataset)

    def _get_label(self, dataset, idx):
        return dataset.__getitem__(idx)['mask']


# class WeightedRandomSampler(Sampler):
#     def __init__(self, dataset, class_weights, replacement=True):
#         self.dataset = dataset
#         self.num_samples = self.__len__()
#         self.indices = list(range(len(dataset)))
#         self.replacement = replacement
#         self.class_weights = class_weights
#         sample_weights_path = os.path.join(os.path.dirname(__file__), 'tmp_storage/sample_weights.pickle')
#         if not os.path.exists(sample_weights_path):
#             weights = [1.0 / self.class_weights[int(self._get_label(dataset, idx).cpu().numpy())]
#                        for idx in self.indices]
#             with open(sample_weights_path, 'wb+') as f:
#                 pickle.dump(weights, f)
#         else:
#             with open(sample_weights_path, 'rb') as f:
#                 weights = pickle.load(f)
#         self.weights = torch.DoubleTensor(weights)
#         print('WeightedRandomSampler upload!')
#
#     def __iter__(self):
#         return iter(torch.multinomial(self.weights, self.num_samples, self.replacement))
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def _get_label(self, dataset, idx):
#         return dataset.__getitem__(idx)['mask']
