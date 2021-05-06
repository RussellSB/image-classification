from hparams import * 

from models import get_model
from log import log_writer

from data import get_dataloaders
from train import train_model
from test import test_model

if __name__ == '__main__':
    # Log hyperparameters, prepare error writer, and load model
    writer = log_writer(logpath, expid, model_str, epochs, batch_size, dataset, lr)
    model = get_model(model_str, num_classes, dataset)

    # Prepare data and train loaded model on data
    trainloader, testloader = get_dataloaders(dataset, datapath, batch_size)
    model = train_model(model, trainloader, testloader, lr, writer, epochs, batch_size, model_str)

    # Test trained model and save results
    test_model(model, testloader, batch_size, logpath, dataset)

