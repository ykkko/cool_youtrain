from albumentations import *
from albumentations.torch import ToTensor
from albumentations import (
    OneOf,
    HorizontalFlip, ShiftScaleRotate,
    GaussNoise, MotionBlur, RandomContrast, RandomBrightness, MedianBlur, ToGray, JpegCompression,
)


def pre_transform(resize):
    transforms = []
    transforms.append(Resize(resize, resize))
    return Compose(transforms)


def post_transform():
    return Compose([
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)),
        ToTensor()])


def mix_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        OneOf([
            GaussNoise(p=.9), MotionBlur(p=.9), MedianBlur(p=.9),
        ], p=.5),
        OneOf([
            RandomContrast(p=.9), RandomBrightness(p=.9),   # ToGray, JpegCompression,
        ], p=.5),
        HorizontalFlip(p=.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=10, interpolation=1, border_mode=4,
                         p=0.6),
        post_transform(),
    ])


def mix_transform2(resize):
    return Compose([
        pre_transform(resize=resize),
        OneOf([
            GaussNoise(p=.9), MotionBlur(p=.9), MedianBlur(p=.9),
        ], p=.6),
        OneOf([
            RandomContrast(p=.9), RandomBrightness(p=.9),
        ], p=.6),
        OneOf([
            ToGray(p=.9),
            JpegCompression(p=.9)
        ], p=.6),
        HorizontalFlip(p=.6),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=10, interpolation=1, border_mode=4,
                         p=0.6),
        post_transform(),
    ])


def test_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        post_transform()]
    )
