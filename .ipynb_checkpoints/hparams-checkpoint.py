import torch

model_str = 'resnet152' # options: ['resnet152', 'vgg19_bn', 'googlenet'] 
expid = '12'
epochs = 10 # 5 
batch_size = 128
dataset = 'FashionMNIST'  # options: ['MNIST', 'CIFAR10', 'FashionMNIST']
num_classes = '10'
lr = 0.05

torch.cuda.set_device(1)
device = 'cuda'

logpath = 'runs/'+expid
datapath = 'data/'
