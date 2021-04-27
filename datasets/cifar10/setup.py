import torchvision.transforms as transforms
import torchvision.datasets as datasets

train_dir = './train/'  # normalised train directory
test_dir = './test/'  # normalised test directory

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
datasets.CIFAR10(root=train_dir, train=True, transform=transform, download=True)  # normalised trainset
datasets.CIFAR10(root=test_dir, train=False, transform=transform, download=True)  # normalised testset  (RUNS ERROR ATM)