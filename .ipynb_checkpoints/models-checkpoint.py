import torchvision.models as models
from hparams import device
from torch import nn

def get_model(model_str, num_classes, dataset):
    '''
    Returns either resnet, vgg or googlenet initialised with random weights.
    If MNIST, modifies the first layer so input is with respect to 1 channel not 3.
    '''
        
    model = eval('models.' + model_str + '(num_classes=' + num_classes + ')')  # example: models.resnet18(num_classes=10)

    # Modify the respective model so that the first layer is wrt 1 channel (B&W) as opposed to 3 (RGB)
    if 'MNIST' in dataset:

        if 'resnet' in model_str:
            # source reference: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adapt to one channel for MNIST

        if 'vgg' in model_str:
            # source reference: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
            layers = list(model.features.children())[:-1]
            layers[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            model.features = nn.Sequential(*layers)

        if 'googlenet' in model_str:
            # source reference: https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html
            from torchvision.models.googlenet import BasicConv2d
            model.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3) 

    model = model.to(device)
    return model

