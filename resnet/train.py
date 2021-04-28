import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
path = '../data/'

import torchvision.models as models
from torch import nn, optim
import torch
torch.cuda.set_device(3)
device = 'cuda'

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torchvision.utils as utils

# =====================================================================================================
#                                       Hyperparameters
# =====================================================================================================

epochs = 10
batch_size = 32
dataset = 'mnist'  # ['mnist', 'cifar10']  TODO (for now just work on mnist)

# =====================================================================================================
#                                       Model Instantiation
# =====================================================================================================
        
resnet = models.resnet18()  # 152 if have time
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adapt to one channel for MNIST
resnet = resnet.to(device)

# =====================================================================================================
#                                       Test and Train Data 
# =====================================================================================================

# Hotfix for Yann Le Cun server being down
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [('/'.join([new_mirror, url.split('/')[-1]]), md5) for url, md5 in datasets.MNIST.resources]

# Loading MNIST data
transform_norm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

trainset = datasets.MNIST(root=path+'train/', train=True, download=True, transform=transform_norm)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

testset = datasets.MNIST(root=path+'test/', train=False, download=True, transform=transform_norm)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

# =====================================================================================================
#                                       Optimization functions
# =====================================================================================================

criterion = nn.CrossEntropyLoss().to(device)  # Cross entropy for multi-class problems
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.005)  # Adam optimizer for momentum-rms balance

# =====================================================================================================
#                                     The Training Loop
# =====================================================================================================
# Epochs loop
pbar_epoch = tqdm(range(epochs), desc='Epochs')
for i in pbar_epoch:
    resnet.train()  # Ensure model is set to training
    
    # Batch loop
    batch_iterations = len(trainloader.dataset)//batch_size
    pbar_batch = tqdm(enumerate(trainloader), leave=False, total=batch_iterations, desc='Batches')
    for j, (x, y) in pbar_batch:
        
        x, y = x.to(device), y.to(device)
        
        # Forward inference and loss computation
        resnet.zero_grad()
        out = resnet(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        # Updating logs
        pbar_epoch.set_postfix(Loss=loss.item())
     