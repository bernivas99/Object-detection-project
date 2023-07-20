import params
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt

def resize_and_convert_coco_bbox(coco_bbox,resize_param_width,resize_param_height):
    x_min=coco_bbox[0]
    y_min=coco_bbox[1] #y_min=coco_bbox[1]-coco_bbox[3]
    x_max=coco_bbox[0]+coco_bbox[2]
    y_max=coco_bbox[1]+coco_bbox[3] #coco_bbox[1]
    
    resized_normal_bbox=[x_min*resize_param_width, y_min*resize_param_height, x_max*resize_param_width, y_max*resize_param_height]
    #new_width=resized_normal_bbox[2]-resized_normal_bbox[0]
    #new_height=resized_normal_bbox[3]-resized_normal_bbox[1]
    #new_coco_coords=[resized_normal_bbox[0],resized_normal_bbox[3],new_width,new_height]
    return resized_normal_bbox

def convert_coco_bbox(coco_bbox):
    x_min=coco_bbox[0]
    y_min=coco_bbox[1] #y_min=coco_bbox[1]-coco_bbox[3]
    x_max=coco_bbox[0]+coco_bbox[2]
    y_max=coco_bbox[1]+coco_bbox[3] #coco_bbox[1]
    
    normal_bbox=[x_min, y_min, x_max, y_max]
    return normal_bbox


def plot_losses(train_losses, val_losses,savefig_name):
    plt.figure()
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.title('Losses during training')
    plt.xlabel("Epoch #")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(params.OUTPUT_DIR, "{}".format(savefig_name)))

def collate_fn(batch):
    return tuple(zip(*batch))



def visualizer(image_path,bbox_cords_list,labels_list,save_image_name):
    #loaded_image = Image.open(image_path).convert("RGB")
    #width, height = loaded_image.size
    #size=(int(width/4), int(height/4))
    #loaded_image = loaded_image.resize(size)
    loaded_image=np.array(image_path.swapaxes(0,2))* 255
    loaded_image=loaded_image.astype(np.uint8)
    loaded_image=Image.fromarray(loaded_image)
    
    classes=['ignored region', 'swimmer', 'floater', 'swimmer on boat', 'floater on boat', 'boat', 'life jacket']
    colors = np.random.uniform(0, 255, size=(params.NUM_OF_CLASSES))
    for i, box in enumerate(bbox_cords_list):
        draw = ImageDraw.Draw(loaded_image)
        
        draw.rectangle([(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))], outline=int(colors[labels_list[i]]),width=1)

        #draw.text((int(box[0]), int(box[1])), classes[labels_list[i]], fill='black',font=ImageFont.truetype('FreeMono.ttf', 65))
    #loaded_image.show()
    loaded_image.save(os.path.join(params.OUTPUT_DIR,'{}.png'.format(save_image_name)))
     