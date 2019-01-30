import numpy as np

import argparse
from pathlib import Path
import glob
import cv2
import pydoc
import torch

from tqdm import tqdm
from dataset import TestDataset
from transforms import test_transform
from torch.utils.data import DataLoader
from albumentations.torch import ToTensor
from youtrain.utils import set_global_seeds, get_config, get_last_save
import torchvision.transforms.functional as F
import pandas as pd
from scipy.misc import imread, imresize
import warnings
warnings.filterwarnings('ignore')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class PytorchInference:
    def __init__(self, device, activation='sigmoid'):
        self.device = device
        self.activation = activation

    @staticmethod
    def to_numpy(images):
        return images.data.cpu().numpy()

    def run_one_predict(self, model, images):
        predictions = model(images)
        if self.activation == 'sigmoid':
            predictions = F.sigmoid(predictions)
        elif self.activation == 'softmax':
            predictions = predictions.exp()
        return predictions

    def predict(self, model, loader):
        model = model.to(self.device).eval()
        with torch.no_grad():
            for data in loader:
                print(type(data))
                print(data.shape)
                images = data.to(self.device)
                predictions = model(images)
                for prediction in predictions:
                    prediction = np.moveaxis(self.to_numpy(prediction), 0, -1)
                    yield prediction

    def predict_on_batch(self, model, batch):
        model = model.to(self.device).eval()
        if isinstance(batch, np.ndarray):
            data = ToTensor()(image=batch)['image'].permute(1, 0, 2, 3)
        with torch.no_grad():
            images = data.to(self.device)
            predictions = model(images)
            for prediction in predictions:
                prediction = np.moveaxis(self.to_numpy(prediction), 0, -1)
                yield prediction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--paths', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config)
    paths = get_config(args.paths)
    params = config['train_params']
    model_name = config['train_params']['model']
    model = pydoc.locate(model_name)(**params['model_params'])
    model.load_state_dict(torch.load(params['weights'])['state_dict'])
    paths = paths['data']

    files = glob.glob('/mnt/hdd1/datasets/naive_data/shot_dataset/test/closeup/*.*')
    batch = np.zeros([16, 224, 224, 3])
    for i in range(batch.shape[0]):
        batch[i, :, :, :] = imresize(imread(files[i]), (224, 224))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferencer = PytorchInference(device)
    preds = inferencer.predict_on_batch(model, batch)
    print(np.array(list(preds)))
    # for i, pred in enumerate(preds):
    #     print(i, ':', pred)
