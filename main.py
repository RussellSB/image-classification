from hparams import * 

from models import get_model
from log import log_writer

from data import get_dataloaders
from train import train_model
from test import test_model

if __name__ == '__main__':
    # Log hyperparameters, prepare error writer, and load model
    writer = log_writer()
    model = get_model(model_str, num_classes, dataset)

    # Prepare data and train loaded model on data
    trainloader, testloader = get_dataloaders(dataset, datapath)
    model = train_model(model, trainloader, testloader, writer)

    # Test trained model and save results
    test_model(model, testloader, dataset)

