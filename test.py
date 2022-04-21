from detectron2.engine import DefaultPredictor

import os,pickle
from utils import *

cfg_save_path = "OD_cfg.pickle"

with open(cfg_save_path,"rb") as f:
    cfg = pickle.load(f)

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
cfg.MODEL.WEIGHTS = "./model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55

predictor = DefaultPredictor(cfg)

image_path = "/home/guju/Desktop/test.png"

on_image(image_path,predictor)