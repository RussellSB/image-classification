import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import torchvision.models as models
from torch import nn, optim
import torch
device = 'cuda'
torch.cuda.set_device(1)

from tqdm import tqdm
from sklearn.metrics import classification_report

import torchvision.utils as utils
from tensorboardX import SummaryWriter
 
import shutil
import os

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

# =====================================================================================================
#                                       Hyperparameters
# =====================================================================================================

model_str = 'vgg19_bn' # ['resnet152', 'vgg19_bn', 'googlenet'] 
expid = '10'
epochs = 10 # 5 
batch_size = 128
dataset = 'FashionMNIST'  # ['MNIST', 'CIFAR10', 'FashionMNIST']
num_classes = '10'
lr = 0.05

# =====================================================================================================
#                                  Progress and Hparam logging
# =====================================================================================================

logpath = 'runs/'+expid

print('Started training for', model_str, 'on', dataset)
if os.path.exists(logpath): print('Overwriting logpath', logpath)
shutil.rmtree(logpath, ignore_errors=True)  # overwrites previous experiment
writer = SummaryWriter(logpath, flush_secs=50)  # tensorboard debugging

f = open(logpath+'/hparams.txt', 'a')  # Opens file for appending hparams 
f.write('expid: ' + expid +'\n')
f.write('model_str: ' + str(model_str) +'\n')
f.write('epochs: ' + str(epochs) +'\n')
f.write('batch_size: ' + str(batch_size) +'\n')
f.write('dataset: ' + dataset +'\n')
f.write('lr: ' + str(lr) +'\n')
f.close()

# =====================================================================================================
#                                       Model Instantiation
# =====================================================================================================
        
model = eval('models.' + model_str + '(num_classes=' + num_classes + ')')  # example: models.resnet18(num_classes=10)

# Modify the respective model so that the first layer is wrt 1 channel (B&W) as opposed to 3 (RGB)
if 'MNIST' in dataset:

    if 'resnet' in model_str:
        # source reference: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adapt to one channel for MNIST

    if 'vgg' in model_str:
        # source reference: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
        layers = list(model.features.children())[:-1]
        layers[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model.features = nn.Sequential(*layers)

    if 'googlenet' in model_str:
        # source reference: https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html
        from torchvision.models.googlenet import BasicConv2d
        model.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3) 

model = model.to(device)


# =====================================================================================================
#                                       Test and Train Data 
# =====================================================================================================

datapath = 'data/'
mean, std = (0.5,), (0.5,)  # To change wrt dataset

if dataset == 'MNIST':
    # Hotfix for Yann Le Cun server being down
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [('/'.join([new_mirror, url.split('/')[-1]]), md5) for url, md5 in datasets.MNIST.resources]
    # Setting dataset params
    mean, std = (0.1307,), (0.3081,)
    datasets = datasets.MNIST
    
if dataset == 'CIFAR10':
    # Setting dataset params
    mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    datasets = datasets.CIFAR10
        
if dataset == 'FashionMNIST':
    # Setting dataset params
    mean, std = (0.2860,), (0.3530,)
    datasets = datasets.FashionMNIST

transform_norm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
trainset = datasets(root=datapath+'train/', train=True, download=True, transform=transform_norm)
testset = datasets(root=datapath+'test/', train=False, download=True, transform=transform_norm)

trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

# =====================================================================================================
#                                       Optimization functions
# =====================================================================================================

criterion = nn.CrossEntropyLoss().to(device)  # Cross entropy for multi-class problems    
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) 

# =====================================================================================================
#                                     The Training Loop
# =====================================================================================================

# Epochs loop
pbar_epoch = tqdm(range(epochs), desc='Epochs')
for i in pbar_epoch:
    
    model.train() # Set to training
    
    # Batch loop
    batch_step = len(trainloader.dataset)//batch_size
    pbar_batch = tqdm(enumerate(trainloader), leave=False, total=batch_step, desc='Batches')
    for j, (x, y) in pbar_batch:
        
        x, y = x.to(device), y.to(device)
        
        # Forward inference and loss computation
        model.zero_grad()
        out = model(x)
        
        '''
        Unlike others, Googlenet implementation returns multiple logit arrays for output, 
        must ensure that it is the relevant first as indicated in the source: 
        
        (https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html)
        
        This must be applied in training, but must not be applied later in testing (for some reason)
        Might be due to some interference from torch.no_grad or model.eval() calls
        '''
        if 'googlenet' in model_str: out = out[0]
            
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        # Updating logs
        step = (i * batch_step) + j
        pbar_epoch.set_postfix(Loss=loss.item())
        writer.add_scalar('data/loss_train', loss.item(), step)
        
    torch.cuda.empty_cache()

# =====================================================================================================
#                                         Evaluation
# =====================================================================================================        
        
# Will store ground truth and predicted    
true_y, pred_y = [], []    
    
model.eval() # Set to evaluating
batch_step = len(testloader.dataset)//batch_size
pbar_test = tqdm(enumerate(testloader), leave=False, total=batch_step, desc='Test Batches')
for i, (x, y) in pbar_test:
    
    x, y = x.to(device), y.to(device)
    
    # Forward inference and loss computation (no_grad greatly minimizes GPU memory consumption)
    with torch.no_grad():
        out = model(x)  
        loss = criterion(out, y)

    # Updating logs
    step = (i * batch_step) + j
    pbar_test.set_postfix(Loss=loss.item())
    writer.add_scalar('data/loss_test', loss.item(), step)
    
    # Get predicted labels based off of likelihood output
    probs = torch.softmax(out, dim=1) 
    preds = torch.argmax(probs, dim=1) 
    
    # Add to prediction comparison buffer
    true_y.extend(y.cpu())
    pred_y.extend(preds.cpu())
         
writer.close()

f = open(logpath+'/results.txt', 'a')  # Opens file for appending test metric results
f.write(classification_report(true_y, pred_y, digits=3))
f.close()

if dataset == 'MNIST':
    classes = ('0', '1', '2', '3', '4','5', '6', '7', '8', '9')
if dataset == 'CIFAR10':
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
if dataset == 'FashionMNIST':
    classes = ('top', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')
    
cf_matrix = confusion_matrix(true_y, pred_y, normalize='true')
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.savefig(logpath+'/confusion_matrix.png')