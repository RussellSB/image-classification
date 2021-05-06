import torch

model_str = 'googlenet' # options: ['resnet152', 'vgg19_bn', 'googlenet'] 
expid = '15'
epochs = 10 # 5 
batch_size = 128
dataset = 'CIFAR10'  # options: ['MNIST', 'CIFAR10', 'FashionMNIST']
num_classes = '10'

max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
optim_func = torch.optim.Adam

torch.cuda.set_device(0)
device = 'cuda'

logpath = 'runs/'+expid
datapath = 'data/'
