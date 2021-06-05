# image-classification

<p align="left">
  Performing and evaluating image classification tasks with deeper networks.
</p>


<p align="center">
  <img width="425" height="225" src="https://github.com/RussellSB/image-classification/blob/main/images/final-results.PNG">
</p>


All training was done using a Quadro RTX 6000. 
All hyperparameters including what model to train are customisable in hparams.py.
This code is only compatible with GPU. If you can only run less expensive models feel free to specify so in hparams.py.
As long as the less expensive model is available from TorchVision Models, it will be retrieved given the model_str specified.

After modifying hparams.py, run the `python main.py`. Note the hparam for experiment ID. This is just for keeping track of your
log output, as well as saving your hyperparameter specifications for large scale testing. 

The dataset can also be specified as required from hparams.py, as long as its provided by TorchVision. 

## Resources

- https://github.com/lukemelas/EfficientNet-PyTorch
- https://pytorch.org/vision/stable/models.html
- https://pytorch.org/vision/stable/datasets.html

## Notes

The deep networks tested include:
- VGG-19
- GoogLeNet
- ResNet-152
- EfficientNet-B3

Tested datasets include:
- MNIST
- FashionMNIST
- CIFAR-10
