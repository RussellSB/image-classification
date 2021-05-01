import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import torchvision.models as models
from torch import nn, optim
import torch
torch.cuda.set_device(0)
device = 'cuda'

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

model_str = 'resnet152' # ['resnet152', ...] (TODO)
expid = '01'
epochs = 5 
batch_size = 64
dataset = 'mnist'  # ['mnist', 'cifar10'] (for now just work on mnist)
num_classes = '10'
lr = 0.005

# =====================================================================================================
#                                  Progress and Hparam logging
# =====================================================================================================

logpath = 'runs/'+expid

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
        
model = eval('models.' + model_str + '(num_classes=' + num_classes + ')')  # example: models.resnet18()

if dataset == 'mnist':  # and resnet
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adapt to one channel for MNIST

model = model.to(device)

# =====================================================================================================
#                                       Test and Train Data 
# =====================================================================================================

datapath = 'data/'

# Hotfix for Yann Le Cun server being down
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [('/'.join([new_mirror, url.split('/')[-1]]), md5) for url, md5 in datasets.MNIST.resources]

# Loading MNIST data
transform_norm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

trainset = datasets.MNIST(root=datapath+'train/', train=True, download=True, transform=transform_norm)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

testset = datasets.MNIST(root=datapath+'test/', train=False, download=True, transform=transform_norm)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

# =====================================================================================================
#                                       Optimization functions
# =====================================================================================================

criterion = nn.CrossEntropyLoss().to(device)  # Cross entropy for multi-class problems
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)  # Adam optimizer for momentum-rms balance

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
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        # Updating logs
        step = (i * batch_step) + j
        pbar_epoch.set_postfix(Loss=loss.item())
        writer.add_scalar('data/loss_train', loss.item(), step)

# =====================================================================================================
#                                         Evaluation
# =====================================================================================================        
        
# Will store ground truth and predicted    
true_y, pred_y = [], []    
    
model.eval() # Set to evaluating
batch_step = len(trainloader.dataset)//batch_size
pbar_test = tqdm(enumerate(testloader), leave=False, total=batch_step, desc='Test Batches')
for i, (x, y) in pbar_test:
    
    x, y = x.to(device), y.to(device)
    
    # Forward inference and loss computation
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

classes = ('0', '1', '2', '3', '4','5', '6', '7', '8', '9')

cf_matrix = confusion_matrix(true_y, pred_y, normalize='true')
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.savefig(logpath+'/confusion_matrix.png')