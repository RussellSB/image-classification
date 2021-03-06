from hparams import device, batch_size, logpath, labels
from train import criterion
from tqdm import tqdm
import torch

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

def test_model(model, testloader, dataset):       
    # Will store ground truth and predicted    
    true_y, pred_y = [], []    

    model.eval() # Set to evaluating
    
    batch_step = len(testloader.dataset)//(batch_size*2)
    pbar_test = tqdm(enumerate(testloader), leave=False, total=batch_step, desc='Testing Batches')
    for i, (x, y) in pbar_test:

        # Forward inference and loss computation (no_grad greatly minimizes GPU memory consumption)
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            out = model(x)  
            loss = criterion(out, y)

        # Updating logs
        pbar_test.set_postfix(Loss=loss.item())

        # Get predicted labels based off of likelihood output
        probs = torch.softmax(out, dim=1) 
        preds = torch.argmax(probs, dim=1) 

        # Add to prediction comparison buffer
        true_y.extend(y.cpu())
        pred_y.extend(preds.cpu())

    f = open(logpath+'/results.txt', 'a')  # Opens file for appending test metric results
    f.write(classification_report(true_y, pred_y, digits=3))
    f.close()

    classes = labels[dataset]

    cf_matrix = confusion_matrix(true_y, pred_y, normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, cmap='viridis')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(logpath+'/confusion_matrix.png')