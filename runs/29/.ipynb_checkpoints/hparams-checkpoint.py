import torch

model_str = 'efficientnet-b3' # options: ['resnet152', 'vgg19_bn', 'googlenet'] ... 'efficientnet-b3'
expid = '31'
epochs = 15
batch_size = 128
dataset = 'CIFAR10'  # options: ['MNIST', 'CIFAR10', 'FashionMNIST']
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