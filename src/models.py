import torch.nn as nn
# from torchvision.models import densenet121

from src.model_zoo.classification.resnet import resnet34
from src.model_zoo.classification.densenet import densenet121
from src.model_zoo.classification.pd_densenet import pd_densenet121
from src.model_zoo.classification.se_pd_densenet import se_pd_densenet121
from src.model_zoo.classification.se_resnet import se_resnet34


def Resnet34_Shot(pretrained, num_classes):
    model = resnet34(pretrained=pretrained)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(512, num_classes, bias=True)
    return model


def SE_Resnet34_Shot(pretrained, num_classes):
    model = se_resnet34(pretrained)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(512, num_classes, bias=True)
    return model


def Densenet121_Shot(pretrained, num_classes):
    model = densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, num_classes, bias=True)
    return model


def Pd_Densenet121_Shot(pretrained, num_classes):
    model = pd_densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, num_classes, bias=True)
    return model

def SE_Pd_Densenet121_Shot(pretrained, num_classes):
    model = se_pd_densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, num_classes, bias=True)
    return model
