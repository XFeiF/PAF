'''
@author: Aeolus
@url: x-fei.me
@time: 2019-04-09 11:21
'''

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo
import torchvision
from torchvision.models import resnet
from src.net_base import *

NUM_BLOCKS = {
    18: [2, 2, 2, 2],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}

class Res18(resnet.ResNet):
    def __init__(self, num_classes=10):
        super(Res18, self).__init__(resnet.BasicBlock, NUM_BLOCKS[18])
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        init_params(self.fc)


class Res50(resnet.ResNet):
    def __init__(self, num_classes=10):
        super(Res50, self).__init__(resnet.Bottleneck, NUM_BLOCKS[50])
        self.expansion = resnet.Bottleneck.expansion
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        init_params(self.fc)

class Res101(resnet.ResNet):
    def __init__(self, num_classes=10):
        super(Res101, self).__init__(resnet.Bottleneck, NUM_BLOCKS[101])
        self.expansion = resnet.Bottleneck.expansion
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet101']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        init_params(self)

class Res152(resnet.ResNet):
    def __init__(self, num_classes=10):
        super(Res152, self).__init__(resnet.Bottleneck, NUM_BLOCKS[152])
        self.expansion = resnet.Bottleneck.expansion
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['res152']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        init_params(self)


def get_net(args):
    model = args['model']
    num_classes = args['num_classes']

    if model == 'res18':
        return Res18(num_classes=num_classes)
    elif model == 'res50':
        return Res50(num_classes=num_classes)
    elif model == 'res101':
        return Res101(num_classes=num_classes)
    elif model == 'res152':
        return Res152(num_classes=num_classes)
    else:
        raise ValueError('No model: {}'.format(model))
