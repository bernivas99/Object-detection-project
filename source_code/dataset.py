import json
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import numpy as np
from PIL import Image
import torch
import albumentations as A
from torchvision.transforms import Resize, functional

import params
from utils import convert_coco_bbox
import utils

#f = open('/home/bernivas/CompVis_project/data/sea_drones_see_detection/annotations/instances_val.json')
#data = json.load(f)
#print(data.keys()) #'info', 'licenses', 'categories', 'images', 'annotations'

  
class SeaDronesDataset(Dataset):
    def __init__(self,file_path_for_dataset,file_path_for_annotations,transforms,width,height):

        self.width=width
        self.height=height
        self.coco_annotation = COCO(annotation_file=file_path_for_annotations)
        self.image_ids = self.coco_annotation.getImgIds()
        self.file_path_for_dataset=file_path_for_dataset
        self.transforms=transforms
        self.num_classes = len(self.coco_annotation.cats) 
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_metadata = self.coco_annotation.loadImgs([image_id])
        
        loaded_image = Image.open(os.path.join(self.file_path_for_dataset,image_metadata[0]['file_name'])).convert("RGB")
        image_width = loaded_image.size[0]
        image_height = loaded_image.size[1]

        
        resize_transform=A.Compose([A.Resize(height=params.HEIGHT, width=params.WIDTH, always_apply=True),
                        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
        

        annotation_ids = self.coco_annotation.getAnnIds(imgIds=[image_id])
        annotations_for_image = self.coco_annotation.loadAnns(annotation_ids)

        object_num_on_image=len(annotations_for_image)

        resize_param_width, resize_param_height = self.width/image_width, self.height/image_height
        bounding_boxes, labels, annot_ids=[],[],[]
        for annotation in annotations_for_image:
            labels.append(annotation['category_id'])
            bounding_boxes.append(annotation['bbox'])
            

        #annot_ids= torch.tensor(annot_ids)
        iscrowd = torch.zeros((object_num_on_image,), dtype=torch.int64)


        #tmp_bounding_boxes=[]
        #for box in bounding_boxes:
                    #tmp_new_box=convert_coco_bbox(box)
                    #tmp_bounding_boxes.append(tmp_new_box)
        #utils.visualizer(functional.to_tensor(np.array(loaded_image)),np.array(tmp_bounding_boxes),
                            #np.array(labels),'loss_ell_tr_elott')



        final_dict_boxes={'image_id': torch.tensor([image_id]),'iscrowd':iscrowd} 

        new_bounding_boxes=[]
        areas=[]
        if self.transforms:
            transformed_item = self.transforms(image = np.array(loaded_image),
                                     bboxes = bounding_boxes,
                                     class_labels = labels)
            
            

            if len(transformed_item['bboxes'])==0:
                resized_item=resize_transform(image = np.array(loaded_image),
                                     bboxes = bounding_boxes,
                                     class_labels = labels)
                for box in resized_item['bboxes']:

                    new_box=convert_coco_bbox(box)
                    new_bounding_boxes.append(new_box)
                    areas.append((new_box[3] - new_box[1]) * (new_box[2] - new_box[0]))
                
                final_dict_boxes['boxes']= torch.tensor(new_bounding_boxes)
                final_dict_boxes['labels'] =torch.tensor(resized_item['class_labels'])
                final_dict_boxes['area'] = torch.tensor(areas)
                final_image = functional.to_tensor(np.array(resized_item['image']))

            else:
            
                for box in transformed_item['bboxes']:

                    new_box=convert_coco_bbox(box)
                    new_bounding_boxes.append(new_box)
                    areas.append((new_box[3] - new_box[1]) * (new_box[2] - new_box[0]))
                
                final_dict_boxes['boxes']= torch.tensor(new_bounding_boxes)
                final_dict_boxes['labels'] =torch.tensor(transformed_item['class_labels'])
                final_dict_boxes['area'] = torch.tensor(areas)
                final_image = functional.to_tensor(np.array(transformed_item['image'])).swapaxes(0,1)
        else:

            resized_item=resize_transform(image = np.array(loaded_image),
                                     bboxes = bounding_boxes,
                                     class_labels = labels)
            for box in resized_item['bboxes']:

                new_box=convert_coco_bbox(box)
                new_bounding_boxes.append(new_box)
                areas.append((new_box[3] - new_box[1]) * (new_box[2] - new_box[0]))

            final_dict_boxes['boxes']= torch.tensor(new_bounding_boxes)
            final_dict_boxes['labels'] =torch.tensor(resized_item['class_labels'])
            final_dict_boxes['area'] = torch.tensor(areas)
            final_image = functional.to_tensor(np.array(resized_item['image']))

        #utils.visualizer(final_image,np.array(final_dict_boxes['boxes']),
                            #np.array(final_dict_boxes['labels']),'loss_ell_datasetben')

        return final_image, final_dict_boxes


