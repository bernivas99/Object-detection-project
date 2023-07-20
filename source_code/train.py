
import torch
import torchvision
import os
import time
from tqdm import tqdm

import params
from params import MODEL_PATH
import utils



def train(config_dict,model,optimizer,train_loader,val_loader):
    
    print('Start training...')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5, patience=config_dict['scheduler_patience'], verbose=True)

    train_losses, val_losses = [], []
    start_time = time.time()
    for e in tqdm(range(config_dict['epochs'])): 

        running_train_loss = 0.0
        model.train()
        for (i, item) in enumerate(train_loader):
            images, targets = item
            images = [img.to(params.DEVICE) for img in images]
            targets = [{k: v.to(params.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_train_loss += losses.item() 

        train_loss=running_train_loss/(i+1)
        train_losses.append(train_loss)  

        if (e+1)%10==0:
            print(f"\rEpoch {e+1}",f"\tLoss {train_loss}\t",sep=' ', end='', flush=True)
        

        with torch.no_grad():
            running_val_loss=0
            for (i,item) in enumerate(val_loader):
                images, targets = item
                images = [img.to(params.DEVICE) for img in images]
                targets = [{k: v.to(params.DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                running_val_loss += losses.item()  
                
            val_loss=running_val_loss/(i+1)
            val_losses.append(val_loss)
            if (e+1)%10==0:
                print(f"Validation loss:{val_loss}")

        scheduler.step(val_loss)

    end_time = time.time()
    print("Total time taken for training the model: {:.2f}s".format(end_time - start_time))

    utils.plot_losses(train_losses, val_losses,'First_plot_losses') 

    torch.save(model, os.path.join(MODEL_PATH,"model_0.pth")) 
