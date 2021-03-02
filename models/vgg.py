from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def create_vgg(num_ic,num_classes,use_pretrained=False,feature_extract=False):
    model_ft = models.vgg16(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    modules =[]
    for i in model_ft.features:
        modules.append(i)
    modules[0] = nn.Conv2d(num_ic, 64, kernel_size=3, padding=1) 
    model_ft.features=nn.Sequential(*modules)


    modules2=[]
    for i in model_ft.classifier:
        modules2.append(i)
    modules2.append(nn.Tanh())
    model_ft.classifier = nn.Sequential(*modules2)
    input_size = 224       
    return model_ft
if __name__=='__main__':
    a = create_vgg(4,6)
    print(a)
    inp = torch.ones((1,4,128,128))
    print(a(inp).shape)

