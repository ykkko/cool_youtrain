"""
Original implementation: https://github.com/kazuto1011/grad-cam-pytorch
"""

from collections import OrderedDict
import os
import argparse
import cv2
import numpy as np
import pydoc

import torch
from torch.nn import functional as F
from albumentations import Resize, Normalize, Compose
from albumentations.torch import ToTensor

from youtrain.utils import set_global_seeds, get_config, get_last_save


class _PropagationBase(object):
    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.image = None

    def _encode_one_hot(self, idx):
        one_hot = torch.Tensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds = self.model(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        print(self.probs)
        self.prob, self.idx = self.probs.sort(0, True)
        print(self.prob)
        return self.prob, self.idx

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(_PropagationBase):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.detach()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()

        return gcam.detach().cpu().numpy()


def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


def prepare_image(frame):
    """
    prepare image for neural network
    :param frame: one raw frame from video
    :return: processed image (transformed for NN input)
    """
    transforms = [
        Resize(224, 224),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)),
        ToTensor()
    ]
    transforms = Compose(transforms)
    return transforms(image=frame)['image']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--paths', type=str, default=None)
    return parser.parse_args()


def main():
    classes = {0: 'long', 1: 'medium', 2: 'closeup', 3: 'detail'}
    topk = len(classes)
    arch = 'resnet34'
    target_layer = 'layer4.2'
    image_path = r'C:\NAIVE\datasets\shot_total_bigger\fixed_train\3_medium\eddi_orel_155.jpg'
    dst_path = r'C:/NAIVE/test_videos/gram_cam/'

    args = parse_args()
    config = get_config(args.config)
    params = config['train_params']
    model_name = config['train_params']['model']
    model = pydoc.locate(model_name)(**params['model_params'])
    model.load_state_dict(torch.load(params['weights'])['state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    raw_image = cv2.imread(image_path)
    image = prepare_image(raw_image)
    image = image.unsqueeze(0)
    image = image.to(device)

    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image)
    idx = idx.cpu().numpy()
    for i in range(0, topk):
        gcam.backward(idx=idx[i])
        output = gcam.generate(target_layer=target_layer)
        save_gradcam(filename=os.path.join(dst_path, '{}_gcam_{}.png'.format(classes[idx[i]], arch)),
                     gcam=output,
                     raw_image=raw_image)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))


if __name__ == '__main__':
    main()
