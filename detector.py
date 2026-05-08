from argparse import ArgumentParser
import os
import numpy as np
import cv2
from rfdetr import RFDETRBase, RFDETRMedium
from PIL import Image
import math
from tqdm import tqdm
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def inference(args):
    model = RFDETRMedium(pretrain_weights=args.model)
    model.optimize_for_inference()
    for seq_id, sequence in tqdm(enumerate(os.listdir(args.dataset_path))):
        if os.path.isdir(f"{args.dataset_path}/{sequence}")==False:
            continue
        if os.path.exists(f"{args.output_path}/{sequence}.txt"):
            print(f"Results for {sequence} already exist.")
            continue
        result_file=open(f"{args.output_path}/{sequence}.txt", "a")
        img_list = sorted(os.listdir(f"{args.dataset_path}/{sequence}/img"), key=lambda x: int(x.split('.')[0]))
        for frame_no, image in tqdm(enumerate(img_list)):
            detections = model.predict(Image.open(f"{args.dataset_path}/{sequence}/img/{image}"), threshold=args.conf_thresh)
            boxes = detections.xyxy
            classes = [class_id for class_id in map(int, detections.class_id)]
            scores = detections.confidence
            for box, class_name, score in zip(boxes, classes, scores):
                print(f"{frame_no} -1 {class_name} -1 -1 -1 {box[0]} {box[1]} {box[2]} {box[3]} -1 -1 -1 -1000 -1000 -1000 -10 1",file = result_file)
        print(f"Processed sequence : {sequence}")

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model path")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the results")
parser.add_argument("--conf_thresh", type=float, default=0.2, help="Confidence threshold")
args = parser.parse_args()
print(f"Using conf thresh of {args.conf_thresh}")
os.makedirs(args.output_path, exist_ok=True)
inference(args)
