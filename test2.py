import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog


from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from detectron2.config import get_cfg
#from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "./train/_annotations.coco.json", "./train")
register_coco_instances("my_dataset_val", {}, "./valid/_annotations.coco.json", "./valid")
register_coco_instances("my_dataset_test", {}, "./test/_annotations.coco.json", "./test")

# my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
# dataset_dicts = DatasetCatalog.get("my_dataset_train")

# import random
# from detectron2.utils.visualizer import Visualizer

# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
    # cv2.imshow('',vis.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
#We are importing our own Trainer Module here to use the COCO validation evaluation during training. Otherwise no validation eval occurs.

# class CocoTrainer(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             os.makedirs("coco_eval", exist_ok=True)
#             output_folder = "coco_eval"

#         return COCOEvaluator(dataset_name, cfg, False, output_folder)

#from .detectron2.tools.train_net import Trainer
#from detectron2.engine import DefaultTrainer
# select from modelzoo here: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-object-detection-baselines

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("my_dataset_train",)
# cfg.DATASETS.TEST = ("my_dataset_val",)

cfg.DATALOADER.NUM_WORKERS = 4
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001


cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05




cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500


# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = CocoTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()

#test evaluation
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# cfg.MODEL.WEIGHTS = "model_final.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
# predictor = DefaultPredictor(cfg)
# evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "my_dataset_test")
# inference_on_dataset(trainer.model, val_loader, evaluator)

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.DATASETS.TEST = ("my_dataset_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_test")
test_metadata.set(thing_classes=["","Block", "Table"])

from time import time
import pytesseract
# from detectron2.utils.visualizer import ColorMode
# import glob

# for imageName in glob.glob('./test/*jpg'):
#     im = cv2.imread(imageName)
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                 metadata=test_metadata, 
#                 scale=1.2
#                     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('',out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

im = cv2.imread('/home/guju/Desktop/test5.png')

st = time()
outputs = predictor(im)
end = time()
v = Visualizer(im[:, :, ::-1],
            metadata=test_metadata,
            scale=1
                )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('',out.get_image()[:, :, ::-1])
cv2.waitKey(0)

# print(outputs["instances"].pred_classes)
coords = outputs["instances"].pred_boxes.tensor.numpy()

# res = pytesseract.image_to_string(im, lang='eng')
# print(res)

# for i in range(len(coords)):
#     img2 = im.copy()
#     img2 = img2[int(coords[i][1]):int(coords[i][3]), int(coords[i][0]):int(coords[i][2])]
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,41)

#     im = cv2.rectangle(im, (int(coords[i][0]), int(coords[i][1])), (int(coords[i][2]), int(coords[i][3])), (255, 255, 255), -1)

#     # res = pytesseract.image_to_string(img2)
#     # print(res)

#     cv2.imshow('',img2)
#     cv2.waitKey(0)

print("infered in ", end - st, " seconds")

cv2.imshow('',im)
cv2.waitKey(0)

# res = pytesseract.image_to_string(im, lang='eng')
# print(res)