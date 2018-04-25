import os
import sys
import random
import math
import re
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("/users/ces/dropbox/repos/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config

import nucleus3

# Directory to save things such as logs and trained model
ROOT_DIR = os.path.abspath('/home/user/ec2-user/histo')
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
DATASET_DIR = os.path.join(ROOT_DIR, "data")

# config
config = nucleus3.NucleusConfig()

WEIGHT_SRC = "imagenet"
NUC_MODEL_PATH = MODEL_DIR + "mask_rcnn_histo.h5"
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "training"

config.display()



# training model initialize
model = modellib.MaskRCNN(TEST_MODE, config,
                          MODEL_DIR)
# initialize weights
init_with = WEIGHT_SRC

try:
    model.load_weights(NUC_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
except:
    pass

# imagenet, nucleus, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)

elif init_with == "nucleus":
    # Load weights trained on Nucleus, but skip layers that
    # are different due to the different number of classes
    #
    model.load_weights(NUC_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
else:
    try:
        model.load_weights(model.find_last()[1], by_name=True)
    except:
        print("oops!")
    
# arg parser
parser = argparse.ArgumentParser(
    description='Mask R-CNN for nuclei counting and segmentation')

parser.add_argument("command",
                    metavar="<command>",
                    help="'train' or 'detect'")
parser.add_argument('dataset', #required=False,
                    default=DATASET_DIR,
                    metavar="/path/to/dataset/",
                    #help='Root directory of the dataset'
                   )
parser.add_argument('weights', #required=False,
                    #default=model.find_last()[1],
                    metavar="/path/to/weights.h5",
                    help="Path to weights .h5 file or 'coco'")
parser.add_argument('logs', #required=False,
                    default=LOGS_DIR,
                    metavar="/path/to/logs/",
                    help='Logs and checkpoints directory (default=logs/)')
parser.add_argument('subset', #required=False,
                    metavar="Dataset sub-directory",
                    help="Subset of dataset to run prediction on")


# training model initialize
model = modellib.MaskRCNN(TEST_MODE, config,
                          MODEL_DIR)

args = parser.parse_args(args=["train",DATASET_DIR,
                               NUC_MODEL_PATH,LOGS_DIR,
                               "stage1_train"]
                        )

# train model
nucleus3.args = args
nucleus3.train(model) # be sure to edit "subset" categories in nucleus3
    
# Save weights
#model_path = os.path.join(MODEL_DIR, "mask_rcnn_histo.h5")
#model.keras_model.save_weights(model_path)
