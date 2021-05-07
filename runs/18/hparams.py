import torch

model_str = 'googlenet' # options: ['resnet152', 'vgg19_bn', 'googlenet'] 
expid = '18'
epochs = 15 #15 #10  #5 
batch_size = 128
dataset = 'CIFAR10'  # options: ['MNIST', 'CIFAR10', 'FashionMNIST']
num_classes = '10'

lr = 0.05
momentum = 0.9
weight_decay = 5e-4
optim_func = torch.optim.SGD

torch.cuda.set_device(1)
device = 'cuda'

logpath = 'runs/'+expid
datapath = 'data/'
