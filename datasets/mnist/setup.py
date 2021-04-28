import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Hotfix for Yann Le Cun server being down
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [('/'.join([new_mirror, url.split('/')[-1]]), md5) for url, md5 in datasets.MNIST.resources]

def load_mnist(train_dir, test_dir, transform):
    '''
    Downloading mnist training set and test set, 
    and computing mean and std of train set and returning it
    '''
    trainset = datasets.MNIST(root=train_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(root=test_dir, train=False, transform=transform, download=True)

    mean = trainset.train_data.float().mean()/255
    std = trainset.train_data.float().std()/255

    print('------------')
    print('Number of train samples: ', len(trainset))
    print('Number of test samples: ', len(testset))
    print('Train Mean:', mean)
    print('Train Std:', std)
    print('------------')
    
    return mean, std

    
if __name__ == "__main__":
    path = './data'
    
    print('============ Downloading default ===============')
    train_dir = path+'/init/train/'
    test_dir = path+'/init/test/'
    transform = transforms.Compose(transforms.ToTensor())
    mean, std = load_mnist(train_dir, test_dir, transform)
    
    '''
    Number of train samples:  60000
    Number of test samples:  10000
    Train Mean: tensor(0.1307)
    Train Std: tensor(0.3081)
    '''
    
    print('============ Downloading normalised ============')
    train_dir = path+'/norm/train/'  # normalised train directory
    test_dir = path+'/norm/test/'  # normalised test directory
    transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    mean, std = load_mnist(train_dir, test_dir, transform_norm)
    
    '''
    Number of train samples:  60000
    Number of test samples:  10000
    Train Mean: tensor(0.1307)
    Train Std: tensor(0.3081)
    '''
    
    
    
    
    


