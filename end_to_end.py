import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import math
from argparse import ArgumentParser
import warnings
import cv2

# --- Suppress common warnings ---
warnings.filterwarnings("ignore")

# --- Assumed Dependencies (placeholders if not found) ---
from rfdetr import RFDETRMedium

from trackers.ocsort_tracker.ocsort import OCSort

from ML_dict import get_tracklet_descriptor, get_physics_resnet18_embedder

SCALE_THRESH = 2.0
def size_matters(box1_xyxy, box2_xyxy):
    x1_1, y1_1, x2_1, y2_1 = box1_xyxy
    x1_2, y1_2, x2_2, y2_2 = box2_xyxy

    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2

    if max(w1, w2) / min(w1, w2) > SCALE_THRESH or max(h1, h2) / min(h1, h2) > SCALE_THRESH:
        return -1
    else:
        return 0

class InMemDescriptorCache:
    def __init__(self, dataset_path, tracker_data_in_mem, emb_model, device):
        self.dataset_path = dataset_path
        self.tracklet_data = tracker_data_in_mem
        self.data = {}
        self.model = emb_model
        self.device = device

    def get_descriptor(self, sequence, track_id):
        track_id = int(track_id)
        descriptor = self.data.get(sequence, {}).get(track_id, None)
        if descriptor is not None:
            return descriptor
        
        tracklet = self.tracklet_data.get(track_id, None)

        frame_folder = os.path.join(self.dataset_path, sequence, "img")
        descriptor = get_tracklet_descriptor(tracklet, self.model, frame_folder, device=self.device)

        if sequence not in self.data:
            self.data[sequence] = {}
        self.data[sequence][track_id] = descriptor
        
        return descriptor

    def get_similarity(self, sequence, track_id_A, track_id_B, threshold=0.9):

        if int(track_id_A) == int(track_id_B):
            return True, 1.0

        desc_A = self.get_descriptor(sequence, track_id_A)
        desc_B = self.get_descriptor(sequence, track_id_B)

        if desc_A is None or desc_B is None:
            return False, 0.0

        similarity = np.dot(desc_A, desc_B)
        
        return similarity >= threshold, similarity

def compute_composite_score(box1_xyxy, box2_xyxy, w_iou=0, w_dist=1, w_scale=1):
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

def update_history(new_id, historic_target_ids):
    if new_id in historic_target_ids: historic_target_ids.remove(new_id)
    historic_target_ids.append(new_id)


def stage_1_detection(det_model, sequence_path, conf_thresh):
    img_folder = os.path.join(sequence_path, "img")
    img_list = sorted(os.listdir(img_folder), key=lambda x: int(x.split('.')[0]))
    detections_in_mem = []
    for frame_no, image_name in enumerate(img_list):
        image_path = os.path.join(img_folder, image_name)
        with Image.open(image_path) as img:
            detections = det_model.predict(img, threshold=conf_thresh)
            for box, class_id, score in zip(detections.xyxy, map(int, detections.class_id), detections.confidence):
                detections_in_mem.append({"frame_id": frame_no, "class_id": class_id, "bbox_xyxy": box, "score": score})
    return detections_in_mem, len(img_list)

def stage_2_tracking(detections_in_mem):
    """Runs OC-SORT on in-memory detections and returns tracks."""        
    tracker = OCSort(det_thresh=0.6)
    
    trks_list = []
    for det in detections_in_mem:
        trks_list.append([
            det["frame_id"], -1, det["class_id"], -1, -1, -1, *det["bbox_xyxy"], -1, -1, -1, -1000, -1000, -1000, -10, 1
        ])
        
    seq_trks = np.array(trks_list, dtype=np.float32)
    min_frame, max_frame = seq_trks[:,0].min(), seq_trks[:,0].max()
    
    mot_results_by_frame = defaultdict(list)
    
    for frame_ind in range(int(min_frame), int(max_frame) + 1):
        dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6:10]
        cates = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2]
        scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,-1]
        assert(dets.shape[0] == cates.shape[0])
        online_targets = tracker.update_public(dets, cates, scores)
        trk_num = online_targets.shape[0]
        boxes = online_targets[:, :4]
        ids = online_targets[:, 4]
        frame_counts = online_targets[:, 6]
        sorted_frame_counts = np.argsort(frame_counts)
        frame_counts = frame_counts[sorted_frame_counts]
        cates = online_targets[:, 5]
        cates = cates[sorted_frame_counts].tolist()
        boxes = boxes[sorted_frame_counts]
        ids = ids[sorted_frame_counts]
        for trk in range(trk_num):
            lag_frame = frame_counts[trk]
            if frame_ind < 2*3 and lag_frame < 0:
                continue
            mot_results_by_frame[int(frame_ind+lag_frame)].append({
                'id': int(ids[trk]),
                'bbox': [boxes[trk][0], boxes[trk][1], boxes[trk][2], boxes[trk][3]]
            })
            
    return mot_results_by_frame

def stage_3_sot_conversion(mot_results_by_frame, sequence, dataset_path, emb_model, device):
    detections_by_track = defaultdict(list)
    for frame_id, dets in mot_results_by_frame.items():
        for det in dets:
            detections_by_track[det['id']].append((frame_id, det['bbox']))
            if not det['bbox']:
                exit()
    

    frame_folder = os.path.join(dataset_path, sequence, "img")
    img_list = sorted(os.listdir(frame_folder), key=lambda x: int(x.split('.')[0]))
    num_frames = len(img_list)
    gt_path = os.path.join(dataset_path, sequence, "groundtruth_rect.txt")
    delimiter = '\t' # NOTE: Change delimiter if needed
    gt = np.loadtxt(gt_path, delimiter=delimiter)
    x, y, w, h = gt[0]
    gt_anchor_xyxy = [x, y, x + w, y + h]

    preds = []; target_id = -1; last_known_bbox = gt_anchor_xyxy; target_history = []
    cache = InMemDescriptorCache(dataset_path, detections_by_track, emb_model, device)
    REACQUISITION_SCORE_THRESHOLD = 0.001
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for frame_idx in range(1, num_frames + 1):
        current_detections = mot_results_by_frame.get(frame_idx, [])

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
                        similarity = cache.get_similarity(sequence, target_id, det['id'])
                        if size_score != -1 and similarity:
                            target_id = det['id']
                            prediction_for_this_frame = det['bbox']
                            found_this_frame = True
                            break
                if found_this_frame:
                    break

            # # continue same track
            track_exists = 0
            if not found_this_frame and current_detections:
                for det in current_detections:
                    if det['id'] == target_id:
                        track_exists = 1
                        size_score = size_matters(last_known_bbox, det['bbox'])
                        # size_score = 1
                        score = compute_composite_score(last_known_bbox, det['bbox'])
                        similarity = cache.get_similarity(sequence, target_id, det['id'])
                        if size_score != -1 and score != -1 and similarity:
                            prediction_for_this_frame = det['bbox']
                            found_this_frame = True
                            break
            
            # 2. If ID lost, try to re-acquire by score matching
            if not found_this_frame and last_known_bbox and current_detections and not track_exists:
                best_score = -1
                best_match = None
                for det in current_detections:
                    # size_score = size_matters(last_known_bbox, det['bbox'])
                    # if size_score != -1 and cache.get_similarity(sequence, target_id, det['id'], threshold=0.9):
                    similarity = cache.get_similarity(sequence, target_id, det['id'])
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
                        preds.append(last_known_bbox)                          # Carry over previous

                    else:
                        ref_crop = img.crop(tuple(last_known_bbox))
                        crops.append(preprocess(ref_crop))
                        bboxes.append(last_known_bbox)

                        inp = torch.stack(crops).to(device)
                        with torch.no_grad():
                            embs = emb_model(inp).cpu()

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

                        if best_score >= 0.9:# and margin > 0.3:
                            preds.append(bboxes[best_idx])                          # Update with best match
                            last_known_bbox = bboxes[best_idx]
                        else:
                            preds.append(last_known_bbox)                           # Carry over previous
    return preds


# --- Main Execution Logic (No changes needed here) ---
def main(args):
    # Main function remains the same as before...
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running on device: {device} ---")

    det_model = RFDETRMedium(pretrain_weights=args.detector)
    det_model.optimize_for_inference()
    
    emb_model_checkpoint = args.physics_emb
    emb_model = get_physics_resnet18_embedder(emb_model_checkpoint, device)

    all_sot_preds = {}; metrics = defaultdict(list)
    sequences = [s for s in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, s))]
    
    import time
    det_runtime=0
    oc_sort_runtime=0
    sot_runtime=0
    total_num_frames=0
    for sequence in tqdm(sequences, desc="Processing Sequences"):
        sequence_path = os.path.join(args.dataset_path, sequence)
        start_time_det = time.time()
        detections_mem, num_frames = stage_1_detection(det_model, sequence_path, conf_thresh=0.2)
        end_time_det = time.time()
        mot_tracks_mem = stage_2_tracking(detections_mem)
        end_time_track = time.time()
        sot_preds = stage_3_sot_conversion(mot_tracks_mem, sequence, args.dataset_path, emb_model, device)
        end_time = time.time()
        

        all_sot_preds[sequence] = sot_preds
        
        det_runtime += (end_time_det - start_time_det)
        oc_sort_runtime += (end_time_track - end_time_det)
        sot_runtime += (end_time - end_time_track)
        total_num_frames+= num_frames

    runtime = det_runtime + oc_sort_runtime + sot_runtime
    mean_fps = total_num_frames /runtime  if runtime > 0 else 0
    print(f"\n--- Total Runtime: {runtime:.2f} seconds for {total_num_frames} frames ---")
    print("\n" + "="*50); print("📊 End-to-End Pipeline Performance Summary 📊")
    print(f"  - Sequences Processed: {len(sequences)}")
    print(f"  - Average End-to-End FPS: {mean_fps:.2f}")
    print(f"Total Detection Time: {det_runtime:.2f} seconds")
    print(f"Total OC-SORT Tracking Time: {oc_sort_runtime:.2f} seconds")
    print(f"Total SOT Conversion Time: {sot_runtime:.2f} seconds")

    output_dir = os.path.join("results", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.draw:
        output_video_dir = os.path.join("results", f"{args.experiment_name}_draw")
        os.makedirs(output_video_dir, exist_ok=True)

    for sequence, preds in all_sot_preds.items():
        with open(os.path.join(output_dir, f"{sequence}.txt"), "w") as f:
            for pred_bbox in preds:
                x1, y1, x2, y2 = pred_bbox; w, h = x2 - x1, y2 - y1
                f.write(f"{int(x1)}\t{int(y1)}\t{int(w)}\t{int(h)}\n")
        
        if args.draw:
            output_video_path = os.path.join(output_video_dir, f"{sequence}.avi")
            img_folder = os.path.join(args.dataset_path, sequence, "img")
            img_list = sorted(os.listdir(img_folder), key=lambda x: int(x.split('.')[0]))
            
            first_img = cv2.imread(os.path.join(img_folder, img_list[0]))
            height, width, _ = first_img.shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
            
            gt_path = os.path.join(args.dataset_path, sequence, "groundtruth_rect.txt")
            try:
                gt = np.loadtxt(gt_path, delimiter='\t')
            except ValueError:
                gt = np.loadtxt(gt_path, delimiter=',')
                
            for i, image_name in enumerate(img_list):
                img_path = os.path.join(img_folder, image_name)
                frame = cv2.imread(img_path)

                # Draw prediction
                if i < len(preds) and preds[i] is not None:
                    x1, y1, x2, y2 = [int(v) for v in preds[i]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red for Prediction
                    cv2.putText(frame, 'PRED', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw ground truth
                if i < len(gt):
                    x, y, w, h = [int(v) for v in gt[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green for Ground Truth
                    cv2.putText(frame, 'GT', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                out.write(frame)
            out.release()
            print(f"  -> Visualization saved to {output_video_path}")

    print(f"Final SOT results have been saved to: '{output_dir}'")

if __name__ == "__main__":
    parser = ArgumentParser(description="An end-to-end pipeline for detection, tracking, and SOT conversion.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the root dataset directory.")
    parser.add_argument("--detector", type=str, required=True, help="Path to the detection model weights (.pth).")
    parser.add_argument("--physics_emb", type=str, required=True, help="Path to the physics tracker model weights (.pth).")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name for the experiment, used as the output folder name.")
    parser.add_argument("--draw", action="store_true", help="Generate and save visualization videos")
    args = parser.parse_args()
    main(args)