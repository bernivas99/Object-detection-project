import numpy as np
from pycocotools.coco import COCO
import os
import torch
from PIL import Image

import params

def predict_and_visualize(model): 

    test_images=[988,983] 

    model.eval()
    colors = np.random.uniform(0, 255, size=(params.NUM_OF_CLASSES, 3)) 
    detection_threshold = 0.8
    total_imgs_iterated = 0
    classes=['ignored region', 'swimmer', 'floater', 'swimmer on boat', 'floater on boat', 'boat', 'life jacket']

    coco_annotation = COCO(annotation_file=os.path.join(params.PARENT_DIR,'annotations/instances_test.json'))
    image_ids = coco_annotation.getImgIds()
    image_id = image_ids[0]
    image_metadata = coco_annotation.loadImgs([image_id])
    loaded_image = Image.open(os.path.join(params.PARENT_DIR,'images/test',image_metadata[0]['file_name'])).convert("RGB")
    loaded_image = torch.tensor(loaded_image).to(params.DEVICE)
    # add batch dimension
    image = torch.unsqueeze(loaded_image, 0)

    with torch.no_grad():
        outputs = model(image) 
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

