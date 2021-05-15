from tensorboardX import SummaryWriter
from hparams import *
import shutil
import os

def log_writer():
    '''
    Logs hyperparameters into text file, and returns writer
    for logging train and test errors via tensorboard
    '''
    print('Started training for', model_str, 'on', dataset)
    if os.path.exists(logpath): 
        print('Going to overwrite', logpath, '... Are you sure?')
        input()
        print('Overwriting logpath', logpath)
    shutil.rmtree(logpath, ignore_errors=True)  # overwrites previous experiment
    if not os.path.exists(logpath): os.mkdir(logpath)
    shutil.copy('hparams.py', logpath)  # backs up hyperparameters for reference

    writer = SummaryWriter(logpath, flush_secs=50)  # tensorboard debugging
    return writer