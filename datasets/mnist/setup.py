import torchvision.transforms as transforms
import torchvision.datasets as datasets

init_dir = './init/' # initial training set without normalisation applied
train_dir = './train/'  # normalised train directory
test_dir = './test/'  # normalised test directory

transform = transforms.Compose(transforms.ToTensor())
trainset = datasets.MNIST(root=init_dir, train=True, transform=transform, download=True)

mean = trainset.train_data.float().mean()/255
std = trainset.train_data.float().std()/255

print('Number of training samples: ', len(trainset))
print('Mean', mean)
print('Std', std)

'''
Normalising with respect to the calculated mean and std of the training dataset
(resource: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457)
'''

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(mean,), std=(std,))])
datasets.MNIST(root=train_dir, train=True, transform=transform, download=True)  # normalised trainset
datasets.MNIST(root=test_dir, train=False, transform=transform, download=True)  # normalised testset