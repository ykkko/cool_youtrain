import torch.nn as nn
import math
from torchvision import models
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import torch
from class_models import se_resnext50, SEBlock, resnext101, se_resnext101, resnext50, resnext101

from torchvision.models import resnet50, densenet121, resnet34, resnet18, resnet152
from pretrainedmodels import nasnetalarge, senet154, se_resnext101_32x4d
from class_models import se_resnext101
import re
import torch
from collections import OrderedDict

import warnings
warnings.filterwarnings('ignore')

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

#
# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#     return model
#
#
# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#     return model
#
#
# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model
#
#
# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return model
#
#
# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#     return model

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class ResnetUnet34(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=True, is_deconv=False):
        super(ResnetUnet34, self).__init__()
        self.num_classes = num_classes
        block = BasicBlock
        layers = [3, 4, 6, 3]
        self.encoder = resnet34(pretrained=pretrained)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)
        conv1 = self.maxpool(conv1)

        conv2 = self.layer1(conv1)
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        conv5 = self.layer4(conv4)
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
        return x_out


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x

class SaltNet(nn.Module):

    def __init__(self, ):
        super(SaltNet, self).__init__()

        self.down1 = nn.Sequential(
            ConvBn2d(  1,  64, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d( 64,  64, kernel_size=3, stride=1, padding=1 ),
        )
        self.down2 = nn.Sequential(
            ConvBn2d( 64, 128, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        self.down3 = nn.Sequential(
            ConvBn2d(128, 256, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        self.down4 = nn.Sequential(
            ConvBn2d(256, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
        )
        self.down5 = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
        )

        self.same = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
        )

        self.up5 = nn.Sequential(
            ConvBn2d(1024,512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
        )

        self.up4 = nn.Sequential(
            ConvBn2d(1024,512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1 ),
        )
        self.up3 = nn.Sequential(
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1 ),
        )
        self.up2 = nn.Sequential(
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(128,  64, kernel_size=3, stride=1, padding=1 ),
        )
        self.up1 = nn.Sequential(
            ConvBn2d(128,  64, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d( 64,  64, kernel_size=3, stride=1, padding=1 ),
        )
        self.feature = nn.Sequential(
            ConvBn2d( 64,  64, kernel_size=1, stride=1, padding=0 ),
        )
        self.logit = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0 )




    def forward(self, input):

        down1 = self.down1(input)
        f     = F.max_pool2d(down1, kernel_size=2, stride=2)#, return_indices=True)
        down2 = self.down2(f)
        f     = F.max_pool2d(down2, kernel_size=2, stride=2)
        down3 = self.down3(f)
        f     = F.max_pool2d(down3, kernel_size=2, stride=2)
        down4 = self.down4(f)
        f     = F.max_pool2d(down4, kernel_size=2, stride=2)
        down5 = self.down5(f)
        f     = F.max_pool2d(down5, kernel_size=2, stride=2)

        f  = self.same(f)

        f = F.upsample(f, scale_factor=2, mode='bilinear')
        #f = F.max_unpool2d(f, i4, kernel_size=2, stride=2)
        f = self.up5(torch.cat([down5, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear')
        f = self.up4(torch.cat([down4, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear')
        f = self.up3(torch.cat([down3, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear')
        f = self.up2(torch.cat([down2, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear')
        f = self.up1(torch.cat([down1, f],1))

        f = self.feature(f)
        #f = F.dropout(f, p=0.5)
        logit = self.logit(f)

        return logit


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.bn = SynchronizedBatchNorm2d(out_channels)


    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x ):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        return x

#
# resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
# resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
# resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
# resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
# resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'

class UNetResNet34(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.


    def load_pretrain(self, pretrain_file):
        self.encoder.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self ):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )# 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  #128
        self.encoder4 = self.resnet.layer3  #256
        self.encoder5 = self.resnet.layer4  #512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512+256, 512, 256)
        self.decoder4 = Decoder(256+256, 512, 256)
        self.decoder3 = Decoder(128+256, 256,  64)
        self.decoder2 = Decoder( 64+ 64, 128, 128)
        self.decoder1 = Decoder(128    , 128,  32)

        self.logit    = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  1, kernel_size=1, padding=0),
        )



    def forward(self, x):
        #batch_size,C,H,W = x.shape

        # mean=[0.485, 0.456, 0.406]
        # std =[0.229, 0.224, 0.225]
        # x = torch.cat([
        #     (x-mean[0])/std[0],
        #     (x-mean[1])/std[1],
        #     (x-mean[2])/std[2],
        # ],1)


        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        e2 = self.encoder2( x)  #; print('e2',e2.size())
        e3 = self.encoder3(e2)  #; print('e3',e3.size())
        e4 = self.encoder4(e3)  #; print('e4',e4.size())
        e5 = self.encoder5(e4)  #; print('e5',e5.size())


        #f = F.max_pool2d(e5, kernel_size=2, stride=2 )  #; print(f.size())
        #f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)#False
        #f = self.center(f)                       #; print('center',f.size())
        f = self.center(e5)

        f = self.decoder5(torch.cat([f, e5], 1))  #; print('d5',f.size())
        f = self.decoder4(torch.cat([f, e4], 1))  #; print('d4',f.size())
        f = self.decoder3(torch.cat([f, e3], 1))  #; print('d3',f.size())
        f = self.decoder2(torch.cat([f, e2], 1))  #; print('d2',f.size())
        f = self.decoder1(f)                      # ; print('d1',f.size())

        #f = F.dropout2d(f, p=0.20)
        logit = self.logit(f)                     #; print('logit',logit.size())
        return logit


#
# def conv3x3(in_, out):
#     return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class DecoderSEBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels))
            # SEBlock(planes=out_channels, reduction=16))

    def forward(self, x):
        return self.block(x)


class DecoderSEBlockV3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels),
            SEBlock(planes=out_channels, reduction=16))

    def forward(self, x):
        return self.block(x)


class SENeXt50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.decoder = nn.ModuleList([
            DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv),
            DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv),
        ])

        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x
    

class NeXt50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnext50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.decoder = nn.ModuleList([
            DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv),
            DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv),
        ])

        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x
    

class NeXt101(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnext101(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.decoder = nn.ModuleList([
            DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv),
            DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv),
        ])

        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x

class NeXt152(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnet152(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.decoder = nn.ModuleList([
            DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv),
            DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv),
        ])

        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x

class Resnet50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnet50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.decoder = nn.ModuleList([
            DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv),
            DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv),
        ])

        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x
    
class Resnet18(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnet18(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.decoder = nn.ModuleList([
            DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv),
            DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv),
        ])

        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x

class SENeXt50WithoutPooling(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.decoder = nn.ModuleList([
            DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv),
            DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv),
        ])

        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=3)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x


class MultiSENeXt50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.decoder = nn.ModuleList([
            DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv),
            DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv),
            DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)])

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, 1)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())
        x_cls = self.avgpool(x)
        x_cls = x_cls.view(x_cls.size(0), -1)
        x_cls = self.fc(x_cls).view(x_cls.size(0))

        x = self.center(self.pool(x))
        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)
        return x, x_cls


class MultiSESENeXt50(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(2048, 1)
        
        self.center = DecoderSEBlockV2(2048, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        # x_cls = self.avgpool(x)
        # x_cls = x_cls.view(x_cls.size(0), -1)
        # x_cls = self.fc(x_cls).view(x_cls.size(0))
        #print(x_cls.shape)

        x = self.center(self.pool(x))
        #print(x.shape)

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))
            #print(x.shape)

        x = self.dec1(x)
        #print(x.shape)
        x = self.dec0(x)
        #print(x.shape)
        x = self.final(x)
        #print(x.shape)
        #print('ok')

        return x#, x_cls


class MultiSESENeXt50_2(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.avgpool = nn.AvgPool2d(3)
        self.fc = nn.Linear(2048, 1)

        self.center = DecoderSEBlockV3(2048, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV3(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV3(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV3(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV3(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV3(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        # x_cls = self.avgpool(x)
        # x_cls = x_cls.view(x_cls.size(0), -1)
        # x_cls = self.fc(x_cls).view(x_cls.size(0))

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x#, x_cls

#
# class MultiSESENeXt101(nn.Module):
#     def __init__(self, num_classes=1, num_filters=16, pretrained=False):
#         super().__init__()
#         self.num_classes = num_classes
#         self.pool = nn.MaxPool2d(2, 2)
#         encoder = resnext101(pretrained=pretrained)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.encoder = nn.ModuleList([
#             nn.Sequential(
#                 encoder.conv1,
#                 encoder.bn1,
#                 encoder.relu,
#                 self.pool),
#             encoder.layer1,
#             encoder.layer2,
#             encoder.layer3,
#             encoder.layer4])
#
#         self.avgpool = nn.AvgPool2d(3)
#         self.fc = nn.Linear(2048, 1)
#
#         self.center = DecoderSEBlockV2(2048, num_filters * 8 * 2, num_filters * 8)
#
#         self.decoder = nn.ModuleList([
#             DecoderSEBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
#             DecoderSEBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
#             DecoderSEBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
#             DecoderSEBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
#         ])
#
#         self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
#         self.dec0 = ConvRelu(num_filters, num_filters)
#         self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
#
#     def forward(self, x):
#         encoder_results = []
#         for stage in self.encoder:
#             x = stage(x)
#             encoder_results.append(x.clone())
#
#         # x_cls = self.avgpool(x)
#         # x_cls = x_cls.view(x_cls.size(0), -1)
#         # x_cls = self.fc(x_cls).view(x_cls.size(0))
#
#         x = self.center(self.pool(x))
#
#         for i, decoder in enumerate(self.decoder):
#             x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))
#
#         x = self.dec1(x)
#         x = self.dec0(x)
#         x = self.final(x)
#
#         return x



class MultiSESENeXt101(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = se_resnext101(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.avgpool = nn.AvgPool2d(3)
        self.fc = nn.Linear(2048, 1)

        self.center = DecoderSEBlockV2(2048, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        # x_cls = self.avgpool(x)
        # x_cls = x_cls.view(x_cls.size(0), -1)
        # x_cls = self.fc(x_cls).view(x_cls.size(0))

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x


class MultiResnet34(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.avgpool = nn.AvgPool2d(3)
        self.fc = nn.Linear(512, 1)

        self.center = DecoderSEBlockV2(512, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x

class MultiResnet18(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnet18(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.avgpool = nn.AvgPool2d(3)
        self.fc = nn.Linear(512, 1)

        self.center = DecoderSEBlockV2(512, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, in_channels=3):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        print(x.size())
        x = self.features.pool0(x)

        x = self.features.denseblock1(x)
        print(x.size())
        x = self.features.transition1(x)

        x = self.features.denseblock2(x)
        print(x.size())
        x = self.features.transition2(x)

        x = self.features.denseblock3(x)
        print(x.size())
        x = self.features.transition3(x)

        x = self.features.denseblock4(x)
        x = self.features.norm5(x)
        print(x.size())

        out = F.relu(x, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(x.size(0), -1)
        out = self.classifier(out)
        return out



def densenet161(pretrained=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model



def densenet121(pretrained=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class MultiSEDensenet161(nn.Module):
        def __init__(self, num_classes=1, num_filters=16, pretrained=True):
            super().__init__()
            self.num_classes = num_classes
            self.pool = nn.MaxPool2d(2, 2)
            encoder = densenet161(pretrained=pretrained)
            self.relu = nn.ReLU(inplace=True)

            self.encoder = nn.ModuleList([
                nn.Sequential(
                    encoder.features.conv0,
                    encoder.features.norm0,
                    encoder.features.relu0,
                ),

                nn.Sequential(
                    encoder.features.pool0,
                    encoder.features.denseblock1, ),
                nn.Sequential(
                    encoder.features.transition1,
                    encoder.features.denseblock2, ),
                nn.Sequential(
                    encoder.features.transition2,
                    encoder.features.denseblock3, ),
                nn.Sequential(
                    encoder.features.transition3,
                    encoder.features.denseblock4,
                    encoder.features.norm5),
            ])

            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(2208, 1)

            self.center = DecoderSEBlockV2(2208, num_filters * 8 * 2, num_filters * 8)

            self.decoder = nn.ModuleList([
                DecoderSEBlockV2(2208 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
                DecoderSEBlockV2(2112 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
                DecoderSEBlockV2(768 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
                DecoderSEBlockV2(384 + num_filters * 2, num_filters * 2 * 2,
                                 num_filters * 2 * 2),
            ])

            self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
            self.dec0 = ConvRelu(num_filters, num_filters)
            self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        def forward(self, x):
            encoder_results = []
            for stage in self.encoder:
                x = stage(x)
                encoder_results.append(x.clone())

            # print(x.size())
            # x_cls = self.avgpool(x)
            # print(x_cls.size())
            # x_cls = x_cls.view(x_cls.size(0), -1)
            # print(x_cls.size())

            # x_cls = self.fc(x_cls).view(x_cls.size(0))

            x = self.center(self.pool(x))

            for i, decoder in enumerate(self.decoder):
                # print(x.size(), encoder_results[-i - 1].size())
                x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

            x = self.dec1(x)
            x = self.dec0(x)
            x = self.final(x)
            return x #, x_cls


class MultiSEDensenet121(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = densenet121(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.features.conv0,
                encoder.features.norm0,
                encoder.features.relu0,
            ),

            nn.Sequential(
                encoder.features.pool0,
                encoder.features.denseblock1, ),
            nn.Sequential(
                encoder.features.transition1,
                encoder.features.denseblock2, ),
            nn.Sequential(
                encoder.features.transition2,
                encoder.features.denseblock3, ),
            nn.Sequential(
                encoder.features.transition3,
                encoder.features.denseblock4,
                encoder.features.norm5),
        ])

        self.avgpool = nn.AvgPool2d(5)
        self.fc = nn.Linear(1024, 1)

        self.center = DecoderSEBlockV2(1024, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        # print(x.size())
        #x_cls = self.avgpool(x)
        # print(x_cls.size())
        #x_cls = x_cls.view(x_cls.size(0), -1)
        # print(x_cls.size())

        #x_cls = self.fc(x_cls).view(x_cls.size(0))

        x = self.center(self.pool(x))

        for i, decoder in enumerate(self.decoder):
            # print(x.size(), encoder_results[-i - 1].size())
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)
        return x #, x_cls

def SENext50(pretrained, num_classes):
    model = se_resnext50(pretrained=pretrained)
    model.fc = nn.Linear(2048, num_classes, bias=True)
    return model

def SENext101(pretrained, num_classes):
    if pretrained:
        model = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
    model.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
    model.last_linear = nn.Linear(2048, num_classes, bias=True)
    return model

def NasNetALarge(pretrained, num_classes):
    if pretrained:
        pretrained = 'imagenet'
    model = nasnetalarge(pretrained=pretrained, num_classes=1000)
    model.last_linear = nn.Linear(4032, num_classes, bias=True)
    return model

def SENet154(pretrained, num_classes):
    if pretrained:
        pretrained = 'imagenet'
    model = senet154(pretrained=pretrained, num_classes=1000)
    model.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
    model.last_linear = nn.Linear(2048, num_classes, bias=True)
    return model
