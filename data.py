import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from hparams import batch_size

def get_dataloaders(dataset, datapath):
    '''
    Returns train, validation, and test loader with respect to the dataset specified.
    Each dataset is normalised with respect to their precomputed mean 
    and standard deviation.
    '''
    
    mean, std = (0.5,), (0.5,)  # To change wrt dataset

    if dataset == 'MNIST':
        # Hotfix for Yann Le Cun server being down
        new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
        datasets.MNIST.resources = [('/'.join([new_mirror, url.split('/')[-1]]), md5) for url, md5 in datasets.MNIST.resources]
        # Setting dataset params
        mean, std = (0.1307,), (0.3081,)
        ds = datasets.MNIST

    if dataset == 'CIFAR10':
        # Setting dataset params
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        ds = datasets.CIFAR10

    if dataset == 'FashionMNIST':
        # Setting dataset params
        mean, std = (0.2860,), (0.3530,)
        ds = datasets.FashionMNIST

    # Data Augmentation
    transform_train = transforms.Compose([transforms.Resize((224, 224)),  # Resize to model input
                                         transforms.RandomCrop(224, padding=28, padding_mode='reflect'),  # Random crop 1/8*width padding
                                         transforms.RandomHorizontalFlip(),  # Flip (vertical would probably reduce performance on MNIST)
                                         transforms.ToTensor(),  # Tensor conversion
                                         transforms.Normalize(mean=mean, std=std)])  # Normalise wrt precomputed
        
    transform_test = transforms.Compose([transforms.Resize((224, 224)),  # Resize to model input
                                         transforms.ToTensor(),  # Tensor conversion
                                         transforms.Normalize(mean=mean, std=std)])  # Normalise wrt precomputed
    
    trainset = ds(root=datapath+'train/', train=True, download=True, transform=transform_train)
    testset = ds(root=datapath+'test/', train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=batch_size*2, shuffle=True)  # twice the batch size, can afford due to no grad
    
    return trainloader, testloader
