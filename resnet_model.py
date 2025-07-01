import torch
from torch import nn
from torch import optim
from torchvision import models,transforms
import torch.nn.functional as F

class Model():
    resnet18 = models.resnet18()

    def set_requires_grad(resnet18):
        for name,param in resnet18.named_parameters():
            print(name,param.requires_grad)


    def set_parameter_fc(resnet18):
        in_features= resnet18.fc.in_features
        resnet18.fc = nn.Linear(in_features,7)
        return resnet18




