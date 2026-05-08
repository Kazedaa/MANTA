from argparse import ArgumentParser
import os
import numpy as np
from collections import defaultdict
import cv2
import torch
import math
from tqdm import tqdm
import torch.nn as nn

from cache import DescriptorCache
from PIL import Image
from torchvision import models, transforms

import warnings
warnings.filterwarnings("ignore")

REACQUISITION_SCORE_THRESHOLD = 0.001
SCALE_THRESH = 2.0

from ML_dict import get_resnet18_embedder, get_physics_resnet18_embedder

# def get_resnet18_embedder(device="cuda"):
#     resnet = models.resnet18(pretrained=True)
#     backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop FC

#     class Embedder(nn.Module):
#         def __init__(self, backbone):
#             super().__init__()
#             self.backbone = backbone
#         def forward(self, x):
#             feat = self.backbone(x)
#             return feat.view(feat.size(0), -1)
    
#     return Embedder(backbone).to(device).eval()

def update_history(new_id, historic_target_ids):
    """Moves the new_id to the end of the list to mark it as most recent."""
    if new_id in historic_target_ids:
        historic_target_ids.remove(new_id)
    historic_target_ids.append(new_id)

def size_matters(box1_xyxy, box2_xyxy):
    x1_1, y1_1, x2_1, y2_1 = box1_xyxy
    x1_2, y1_2, x2_2, y2_2 = box2_xyxy

    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2

    if max(w1, w2) / min(w1, w2) > SCALE_THRESH or max(h1, h2) / min(h1, h2) > SCALE_THRESH:
        return -1
    else:
        return 0

def compute_composite_score(box1_xyxy, box2_xyxy, w_iou=0, w_dist=1, w_scale=1):
    """
    Computes a composite score based on IoU, center distance, and scale similarity.
    Higher is better.

    Args:
        box1_xyxy (list): The first bounding box in [x1, y1, x2, y2] format.
        box2_xyxy (list): The second bounding box in [x1, y1, x2, y2] format.
        w_iou (float): Weight for the IoU score.
        w_dist (float): Weight for the distance score.
        w_scale (float): Weight for the scale score.

    Returns:
        float: The final composite score, typically between 0 and 1.
    """
    # --- Basic Box Properties ---
    x1_1, y1_1, x2_1, y2_1 = box1_xyxy
    x1_2, y1_2, x2_2, y2_2 = box2_xyxy

    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2
    
    area1 = w1 * h1
    area2 = w2 * h2

    if area1 <= 0 or area2 <= 0:
        return -1

    # --- 1. IoU Score ---
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = area1 + area2 - intersection_area
    iou_score = intersection_area / union_area if union_area > 0 else 0.0

    # --- 2. Distance Score ---
    cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
    cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
    center_dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    # Normalize distance by the diagonal of the larger box
    diag = math.sqrt(w1**2 + h1**2)
    dist_score = max(0, 1 - (center_dist / diag)) if diag > 0 else 0.0

    # --- 3. Scale Score ---
    scale_score = min(area1, area2) / max(area1, area2)

    # reject if there is absolutely no overlap
    if iou_score <= 0.001:
        return -1
    
    # --- Final Composite Score ---
    composite_score = (w_iou * iou_score + 
                       w_dist * dist_score + 
                       w_scale * scale_score)

    return composite_score

def inference(args):
    cache = DescriptorCache(args.cache_path, args.dataset_path, args.tracker_output)
    for sequence in os.listdir(args.dataset_path):
        # if sequence != "GarryFish":
        #     continue
        if not os.path.isdir(f"{args.dataset_path}/{sequence}"):
            continue
        gt_path = f"{args.dataset_path}/{sequence}/groundtruth_rect.txt"
        tracker_output_path = f"{args.tracker_output}/{sequence}.txt"
        result_path_file = open(f"{args.output_path}/{sequence}.txt", "a")
        frame_folder = f"{args.dataset_path}/{sequence}/img"

        delimiter = '\t' if args.dataset not in ["webuot1m", "uwcot220"] else ','
        gt = np.loadtxt(gt_path, delimiter=delimiter)
        mot_raw = np.loadtxt(tracker_output_path, delimiter=',')

        detections_by_frame = defaultdict(list)
        for row in mot_raw:
            frame_id = int(row[0])
            track_id = int(row[1])
            bbox = [float(x) for x in row[6:10]] # x1, y1, x2, y2
            detections_by_frame[frame_id].append({'id': track_id, 'bbox': bbox})


        img_list = sorted(os.listdir(f"{args.dataset_path}/{sequence}/img"), key=lambda x: int(x.split('.')[0]))
        num_frames = len(img_list)
        preds = []

        # Convert first GT from [x, y, w, h] to [x1, y1, x2, y2] for initialization
        gt_xywh = gt[0]
        gt_anchor_xyxy = [gt_xywh[0], gt_xywh[1], gt_xywh[0] + gt_xywh[2], gt_xywh[1] + gt_xywh[3]]

        target_id = -1
        last_known_bbox = gt_anchor_xyxy

        target_history = []
        track_birth = {}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_physics_resnet18_embedder(args.,device)
        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        for frame_idx in tqdm(range(1, num_frames + 1), desc=f"{sequence}"):
            current_detections = detections_by_frame.get(frame_idx, [])
            # for det in current_detections:
            #     if det['id'] not in track_birth:
            #         track_birth[det['id']] = frame_idx

            # State 1: INITIALIZATION (We haven't found our target yet)
            if target_id == -1:
                best_score = -1
                best_detection = None
                for det in current_detections:
                    score = compute_composite_score(gt_anchor_xyxy, det['bbox'])
                    if score > best_score:
                        best_score = score
                        best_detection = det

                if best_detection and best_score > REACQUISITION_SCORE_THRESHOLD:
                    target_id = best_detection['id']
                    update_history(target_id, target_history)
                    last_known_bbox = gt_anchor_xyxy #best_detection['bbox']
                    preds.append(last_known_bbox)
                else:
                    # target_id = -2
                    preds.append(last_known_bbox) # Carry over previous

            
            # State 2: TRACKING (We have a target_id and are following it)
            else:
                prediction_for_this_frame = None
                found_this_frame = False

                # keep checking if an old track id reappears
                for hist_id in reversed(target_history[:-1]):
                    for det in current_detections:
                        if det['id'] == hist_id:
                            size_score = size_matters(gt_anchor_xyxy, det['bbox'])
                            similarity = cache.get_similarity(sequence, target_id, det['id'])[0]
                            if size_score != -1 and similarity:
                                target_id = det['id']
                                prediction_for_this_frame = det['bbox']
                                found_this_frame = True
                                break
                    if found_this_frame:
                        break

                # continue same track
                track_exists = 0
                if not found_this_frame and current_detections:
                    for det in current_detections:
                        if det['id'] == target_id:
                            track_exists = 1
                            size_score = size_matters(last_known_bbox, det['bbox'])
                            # size_score = 1
                            score = compute_composite_score(last_known_bbox, det['bbox'])
                            similarity = cache.get_similarity(sequence, target_id, det['id'])[0]
                            if size_score != -1 and score != -1 and similarity:
                                prediction_for_this_frame = det['bbox']
                                found_this_frame = True
                                break
                
                # 2. If ID lost, try to re-acquire by score matching
                if not found_this_frame and last_known_bbox and current_detections and not track_exists:
                    best_score = -1
                    best_match = None
                    for det in current_detections:
                        similarity = cache.get_similarity(sequence, target_id, det['id'])[0]
                        if similarity:
                            score = compute_composite_score(last_known_bbox, det['bbox'])
                            if score > best_score:
                                best_score = score
                                best_match = det

                    if best_match and best_score > REACQUISITION_SCORE_THRESHOLD:
                        target_id = best_match['id'] # Re-acquired! Update ID
                        prediction_for_this_frame = best_match['bbox']
                        found_this_frame = True
                
                # 3. Final decision: update or carry over
                if found_this_frame:
                    last_known_bbox = prediction_for_this_frame
                    preds.append(prediction_for_this_frame)
                    update_history(target_id, target_history)
                else:
                    def getCropsfromBbox(img, last_known_bbox):
                        W, H = img.size
                        stride_factor = 0.5
                    
                        xa, ya, xb, yb = [int(k) for k in last_known_bbox]

                        wa = xb - xa
                        ha = yb - ya
                        cx, cy = (xa + xb) // 2, (ya + yb) // 2

                        stride_x, stride_y = int(wa * stride_factor), int(ha * stride_factor)

                        crops = []
                        bboxes = []

                        search_factor = 1.5  # how many box widths/heights to search in each direction

                        half_w = wa / 2
                        half_h = ha / 2

                        for dx in range(int(-wa * search_factor), int(wa * search_factor) + 1, stride_x):
                            for dy in range(int(-ha * search_factor), int(ha * search_factor) + 1, stride_y):
                                # candidate center
                                new_cx = cx + dx
                                new_cy = cy + dy

                                # derive box from center
                                x1 = max(0, int(round(new_cx - half_w)))
                                y1 = max(0, int(round(new_cy - half_h)))
                                x2 = min(W, x1 + wa)
                                y2 = min(H, y1 + ha)

                                # skip degenerate boxes
                                if (x2 - x1) != wa or (y2 - y1) != ha:
                                    continue

                                crop = img.crop((x1, y1, x2, y2))
                                crops.append(preprocess(crop))
                                bboxes.append((x1, y1, x2, y2))

                        return crops, bboxes

                    cosine = torch.nn.CosineSimilarity(dim=1)

                    img = Image.open(os.path.join(frame_folder, img_list[frame_idx-1])).convert("RGB")
                    crops, bboxes = getCropsfromBbox(img, last_known_bbox)

                    if not crops:
                        preds.append(last_known_bbox)                   # Carry over previous

                    else:
                        ref_crop = img.crop(tuple(last_known_bbox))
                        crops.append(preprocess(ref_crop))
                        bboxes.append(last_known_bbox) 

                        inp = torch.stack(crops).to(device)
                        with torch.no_grad():
                            embs = model(inp).cpu()

                        search_embs = embs[:-1]
                        if target_id < 0:
                            last_emb_t = embs[-1].unsqueeze(0)
                        else:
                            last_emb_t = torch.Tensor(cache.get_descriptor(sequence, target_id))

                        scores = cosine(search_embs, last_emb_t)

                        top_scores = torch.topk(scores, 2).values
                        best_score = top_scores[0].item()
                        second_best_score = top_scores[1].item()
                        margin = best_score - second_best_score

                        best_idx = torch.argmax(scores).item()
                        best_score = scores[best_idx].item()

                        if best_score >= 0.9:
                            preds.append(bboxes[best_idx])                  # Update with best match
                            last_known_bbox = bboxes[best_idx]
                        else:
                            preds.append(last_known_bbox)                   # Carry over previous


        # --- Write results to file ---
        for pred_bbox in preds:
            x1, y1, x2, y2 = pred_bbox
            w, h = x2 - x1, y2 - y1
            result_path_file.write(f"{int(x1)}\t{int(y1)}\t{int(w)}\t{int(h)}\n")
        result_path_file.close()

        if args.draw:
            output_video_path = f"{args.output_path}_draw/{sequence}.mp4"
            first_img = cv2.imread(f"{args.dataset_path}/{sequence}/img/{img_list[0]}")
            height, width, _ = first_img.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

            for i, image_name in enumerate(img_list):
                img_path = f"{args.dataset_path}/{sequence}/img/{image_name}"
                frame = cv2.imread(img_path)

                # Draw prediction
                if i < len(preds) and preds[i] is not None:
                    x1, y1, x2, y2 = [int(v) for v in preds[i]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red for Prediction
                    cv2.putText(frame, 'PRED', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    print("No prediction for frame", i)

                # Draw ground truth
                if i < len(gt):
                    x, y, w, h = [int(v) for v in gt[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green for Ground Truth
                    cv2.putText(frame, 'GT', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                out.write(frame)
            out.release()
            print(f"  -> Visualization saved to {output_video_path}")

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the results")
parser.add_argument("--tracker_output", type=str, required=True, help="Path to save the tracker results")
parser.add_argument("--cache_path", type=str, required=True, help="Path to the similarity cache")
parser.add_argument("--draw", action="store_true", help="Generate and save visualization videos")


args = parser.parse_args()
os.makedirs(args.output_path, exist_ok=True)
if args.draw:
    os.makedirs(f"{args.output_path}_draw", exist_ok=True)
inference(args)