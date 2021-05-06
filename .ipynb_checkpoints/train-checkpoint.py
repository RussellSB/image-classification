from torch import nn, optim
from hparams import *
import torch

from tqdm import tqdm

# To be imported to test also
criterion = nn.CrossEntropyLoss().to(device)  # Cross entropy for multi-class problems  

def train_model(model, trainloader, testloader, writer):
    '''
    Defines optimizer and error criterion and trains the model
    on dataset. Two progress bars for epochs and batches respectively
    are displayed.
    '''
    
    #  Optimization function and learning rate scheduling with a Once Cycle Learning Rate Policy
    optimizer = optim_func(model.parameters(), max_lr, weight_decay=weight_decay) 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(trainloader))
    train_step = 0  # step counter, increment with each weight update

    # Epochs loop
    pbar_epoch = tqdm(range(epochs), desc='Epochs')
    for i in pbar_epoch:

        # ============================== Train Batch loop ==============================
        model.train()  # Set to training
        
        batch_steps = len(trainloader.dataset)//batch_size
        pbar_batch = tqdm(enumerate(trainloader), leave=False, total=batch_steps, desc='Training Batches')
        for j, (x, y) in pbar_batch:

            x, y = x.to(device), y.to(device)

            # Forward inference and loss computation
            out = model(x)

            '''
            Unlike others, Googlenet implementation returns multiple logit arrays for output, 
            must ensure that it is the relevant first as indicated in the source: 

            (https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html)

            This must be applied in training, but must not be applied later in testing (for some reason)
            Might be due to some interference from torch.no_grad or model.eval() calls
            '''
            if 'googlenet' in model_str: out = out[0]

            loss = criterion(out, y)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()

            # Updating logs
            pbar_epoch.set_postfix(Loss=loss.item())
            writer.add_scalar('data/loss_train', loss.item(), train_step)
            train_step += 1

        torch.cuda.empty_cache()
        
        # ============================== Validation Batch loop ==============================
        model.eval()  # Validate after each epoch not batch (a relatively inexpensive computation)
        avg_loss = 0  # Average loss
        
        testbatch_steps = len(testloader.dataset)//(batch_size*2)
        pbar_batch = tqdm(enumerate(testloader), leave=False, total=testbatch_steps, desc='Validating Batches')
        for j, (x, y) in pbar_batch:

            x, y = x.to(device), y.to(device)

            # Forward inference and loss computation (no_grad greatly minimizes GPU memory consumption)
            with torch.no_grad():
                model.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                avg_loss += loss.item() / testbatch_steps  # Update average loss

            # Updating logs
            pbar_epoch.set_postfix(Loss=loss.item())
            
        writer.add_scalar('data/loss_val', avg_loss, i)
        
    return model