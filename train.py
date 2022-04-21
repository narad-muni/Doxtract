from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os,pickle

from utils import *

config_file_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
checkpoint_url = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"

output_dir = "./output/object_detection"
num_class = 2

device = "cpu"

train_dataset = "LP_train"
train_images_dir = "./train"
train_json_annot_path = "./train/_annotations.coco.json"

test_dataset = "LP_test"
test_images_dir = "./test"
test_json_annot_path = "./test/_annotations.coco.json"

cfg_save_path = "OD_cfg.pickle"

register_coco_instances(name=train_dataset,metadata={},image_root=train_images_dir,json_file=train_json_annot_path)
register_coco_instances(name=test_dataset,metadata={},image_root=test_images_dir,json_file=test_json_annot_path)

# plot_samples(train_dataset,n=1)

def main():
    cfg = get_train_cfg(config_file_path,checkpoint_url,train_dataset,test_dataset,num_class,device,output_dir)

    with open(cfg_save_path,"wb") as f:
        pickle.dump(cfg,f,protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)    

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()