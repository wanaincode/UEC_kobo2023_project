# Reference
# https://github.com/facebookresearch/segment-anything/tree/main
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py

import torch
import cv2
import json
from tqdm import tqdm
import sys, os
sys.path.append('./segment-anything/segment_anything')
sys.path.append('./segment-anything/')
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import mask_to_rle_pytorch

sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

file_name = "./test.jpg"

image = cv2.imread(file_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# get from object detector
boxes = [[0, 0, 0, 0], [0, 0, 0, 0,]]
input_boxes = torch.tensor(boxes, device=predictor.device)
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)
