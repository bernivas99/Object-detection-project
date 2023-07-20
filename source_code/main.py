import random
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import params
from train import train
from utils import collate_fn,visualizer
from predict import predict_and_visualize
from dataset import SeaDronesDataset
from torchvision.models.detection.faster_rcnn import FasterRCNN 
from torchvision.models.detection.anchor_utils import AnchorGenerator 
from torchvision.models import resnet50, ResNet50_Weights 

config_list=[
    {'epochs':50, 'learning_rate':0.001, 'batch_size':32,'weight_decay':0.1,'scheduler_patience':5},
    #{'epochs':8, 'learning_rate':0.01, 'batch_size':32,'weight_decay':0.1,'scheduler_patience':5}

]

random.seed(42)
torch.manual_seed(params.RANDOM_STATE)
np.random.seed(params.RANDOM_STATE)


train_transforms = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Resize(height=params.HEIGHT, width=params.WIDTH, always_apply=True),
    ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco',min_area=4, min_visibility=0.1,  label_fields=['class_labels'])) #min_area=4, min_visibility=0.1,
    
val_transforms = A.Compose([
    A.Resize(height=params.HEIGHT, width=params.WIDTH, always_apply=True),
    ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco',min_area=4, min_visibility=0.1,  label_fields=['class_labels'])) 


train_dataset=SeaDronesDataset(os.path.join(params.PARENT_DIR,'images/train'),
                            os.path.join(params.PARENT_DIR,'annotations/instances_train.json'),
                            train_transforms, params.WIDTH,params.HEIGHT)
val_dataset=SeaDronesDataset(os.path.join(params.PARENT_DIR,'images/val'),
                            os.path.join(params.PARENT_DIR,'annotations/instances_val.json'),
                            val_transforms, params.WIDTH,params.HEIGHT)


#vmi=train_dataset.__getitem__(2)
#visualizer(vmi[0],np.array(vmi[1]['boxes']),np.array(vmi[1]['labels']),'loss_ell_dataset_kivul')


for j in range(len(config_list)):

    train_loader = DataLoader(train_dataset, shuffle=True,
        batch_size=config_list[j]['batch_size'], num_workers=os.cpu_count(), collate_fn=collate_fn)                                  
    val_loader = DataLoader(val_dataset, shuffle=False,
        batch_size=config_list[j]['batch_size'], num_workers=os.cpu_count(), collate_fn=collate_fn)


    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    modules = list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048
    model = FasterRCNN(backbone=backbone,
                       rpn_anchor_generator=anchor_generator,
                       num_classes=train_dataset.num_classes)



    for name, param in model.named_parameters():
        param.requires_grad = True

    model.to(params.DEVICE)

    prms = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.Adam(prms, lr=config_list[j]['learning_rate'],weight_decay=config_list[j]['weight_decay'])
    optimizer=torch.optim.SGD(prms, lr=config_list[j]['learning_rate'], weight_decay=config_list[j]['weight_decay'])

    train(config_list[j],model,optimizer,train_loader,val_loader)

    model=torch.load(os.path.join(params.MODEL_PATH,"model_0.pth")) 
    model.to(params.DEVICE)

    
    