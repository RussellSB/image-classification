from tensorboardX import SummaryWriter
import shutil
import os

def log_writer(logpath, expid, model_str, epochs, batch_size, dataset, lr):
    '''
    Logs hyperparameters into text file, and returns writer
    for logging train and test errors via tensorboard
    '''
    print('Started training for', model_str, 'on', dataset)
    if os.path.exists(logpath): print('Overwriting logpath', logpath)
    shutil.rmtree(logpath, ignore_errors=True)  # overwrites previous experiment

    writer = SummaryWriter(logpath, flush_secs=50)  # tensorboard debugging

    f = open(logpath+'/hparams.txt', 'a')  # Opens file for appending hparams 
    f.write('expid: ' + expid +'\n')
    f.write('model_str: ' + str(model_str) +'\n')
    f.write('epochs: ' + str(epochs) +'\n')
    f.write('batch_size: ' + str(batch_size) +'\n')
    f.write('dataset: ' + dataset +'\n')
    f.write('lr: ' + str(lr) +'\n')
    f.close()
    
    return writer