from torch import nn, optim
from hparams import device
import torch

from tqdm import tqdm

# To be imported to test also
criterion = nn.CrossEntropyLoss().to(device)  # Cross entropy for multi-class problems  

def train_model(model, trainloader, valoader, lr, writer, epochs, batch_size, model_str):
    '''
    Defines optimizer and error criterion and trains the model
    on dataset. Two progress bars for epochs and batches respectively
    are displayed.
    '''
    
    #  Optimization function  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) 
    
    train_step = 0  # step counter, increment with each weight update

    # Epochs loop
    pbar_epoch = tqdm(range(epochs), desc='Epochs')
    for i in pbar_epoch:

        model.train()  # Set to training

        # Train Batch loop
        batch_steps = len(trainloader.dataset)//batch_size
        pbar_batch = tqdm(enumerate(trainloader), leave=False, total=batch_steps, desc='Training Batches')
        for j, (x, y) in pbar_batch:

            x, y = x.to(device), y.to(device)

            # Forward inference and loss computation
            model.zero_grad()
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
            optimizer.step()

            # Updating logs
            pbar_epoch.set_postfix(Loss=loss.item())
            writer.add_scalar('data/loss_train', loss.item(), train_step)
            train_step += 1

        torch.cuda.empty_cache()
        
        model.eval()  # Validate after each epoch not batch (a relatively inexpensive computation)
        avg_loss = 0  # Average loss
        
        # Validation Batch loop
        batch_steps = len(trainloader.dataset)//batch_size
        pbar_batch = tqdm(enumerate(trainloader), leave=False, total=batch_steps, desc='Validating Batches')
        for j, (x, y) in pbar_batch:

            x, y = x.to(device), y.to(device)

            # Forward inference and loss computation (no_grad greatly minimizes GPU memory consumption)
            with torch.no_grad():
                model.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                avg_loss += loss.item() / batch_steps  # Update average loss

            # Updating logs
            pbar_epoch.set_postfix(Loss=loss.item())
            
        writer.add_scalar('data/loss_val', avg_loss, i)
        
    return model