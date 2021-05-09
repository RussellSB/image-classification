import torch

model_str = 'resnet152' # options: ['resnet152', 'vgg19_bn', 'googlenet'] 
expid = '24'
epochs = 15 #15 #10  #5 
batch_size = 128
dataset = 'FashionMNIST'  # options: ['MNIST', 'CIFAR10', 'FashionMNIST']
num_classes = '10'

lr = 0.05
momentum = 0.9
weight_decay = 5e-4
optim_func = torch.optim.SGD

torch.cuda.set_device(0)
device = 'cuda'

logpath = 'runs/'+expid
datapath = 'data/'

labels = {
    'MNIST': ('0', '1', '2', '3', '4','5', '6', '7', '8', '9'),
    'CIFAR10': ('airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck'),
    'FashionMNIST': ('top', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')
}