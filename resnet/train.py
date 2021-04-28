import torchvision.transforms as transforms
import torchvision.datasets as datasets
path = '../data/'

import torchvision.models as models
from torch import nn, optim
import torch
torch.cuda.set_device(3)
device = 'cuda'

from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torchvision.utils as utils

# ================================== MODEL =================================================================== #

class ResNetGRAY(ResNet):
    '''
    Working with resnet152 (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
    Overwriting from 3 channels to 1 at the beginning so that it can work with MNIST
    '''
    def __init__(self):
        super(ResNetGRAY, self).__init__(Bottleneck, [3, 8, 36, 3], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
        
resnet = ResNetGRAY().to(device)
        

# ================================== DATA ==================================================================== #

# Hotfix for Yann Le Cun server being down
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [('/'.join([new_mirror, url.split('/')[-1]]), md5) for url, md5 in datasets.MNIST.resources]

# Loading MNIST data
transform_norm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
trainset = datasets.MNIST(root=path+'train/', train=True, download=True, transform=transform_norm)
testset = datasets.MNIST(root=path+'test/', train=False, download=True, transform=transform_norm)

# ================================== LOSS ==================================================================== #

# ================================== TRAIN ==================================================================== #