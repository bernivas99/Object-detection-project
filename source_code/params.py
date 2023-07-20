import torch

WIDTH=224
HEIGHT=224 
PARENT_DIR='/home/bernivas/CompVis_project/data/sea_drones_see_detection'
RANDOM_STATE=322
DEVICE="cuda" if torch.cuda.is_available() else "cpu"


# classes: ['ignored region', 'swimmer', 'floater', 'swimmer on boat', 'floater on boat', 'boat', 'life jacket']
NUM_OF_CLASSES=7 

OUTPUT_DIR='/home/bernivas/CompVis_project/output'
MODEL_PATH='/home/bernivas/CompVis_project/output'

