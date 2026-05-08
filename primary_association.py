from loguru import logger
import time

import sys
sys.path.append('./')
from trackers.ocsort_tracker.ocsort import OCSort
from utils.args import make_parser
import os
import motmetrics as mm
import numpy as np
from tqdm import tqdm

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(args):
    results_folder = args.out_path
    raw_path = args.raw_results_path
    os.makedirs(results_folder, exist_ok=True)

    total_time = 0 
    total_frame = 0 

    test_seqs = os.listdir(args.raw_results_path)
    cats = [str(i) for i in range(90)]
    cat_ids = {cat: i for i, cat in enumerate(cats)}

    for seq_name in test_seqs:
        seq_name = seq_name.split('.')[0]
        print("starting seq {}".format(seq_name))
        # if seq_name != "GarryFish":
        #     continue
        tracker = OCSort(args.track_thresh)
        seq_trks = np.empty((0, 18))
        seq_file = os.path.join(raw_path, "{}.txt".format(seq_name))
        seq_file = open(seq_file)
        out_file = os.path.join(results_folder, "{}.txt".format(seq_name))
        if os.path.exists(out_file):
            print(f"Output file {out_file} already exists, skipping.")
            continue
        out_file = open(out_file, 'w')
        lines = seq_file.readlines()
        line_count = 0 
        for line in tqdm(lines):
            line_count+=1
            line = line.strip()
            tmps = line.strip().split()
            tmps[2] = cat_ids[tmps[2]]
            trk = np.array([float(d) for d in tmps])
            trk = np.expand_dims(trk, axis=0)
            seq_trks = np.concatenate([seq_trks, trk], axis=0)
        min_frame = seq_trks[:,0].min()
        max_frame = seq_trks[:,0].max()
        for frame_ind in tqdm(range(int(min_frame), int(max_frame)+1)):
            dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6:10]
            cates = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2]
            scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,-1]

            assert(dets.shape[0] == cates.shape[0])
            t0 = time.time()
            online_targets = tracker.update_public(dets, cates, scores)
            t1 = time.time()
            total_frame += 1
            total_time += t1 - t0
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
                if frame_ind < 2*args.min_hits and lag_frame < 0:
                    continue
                out_line = "{},{},{},-1,-1,-1,{},{},{},{},-1,-1,-1,-1000,-1000,-1000,-10,1\n".format\
                    (int(frame_ind+lag_frame), int(ids[trk]), cates[trk], 
                    boxes[trk][0], boxes[trk][1], boxes[trk][2], boxes[trk][3])
                out_file.write(out_line)

    print("Running over {} frames takes {}s. FPS={}".format(total_frame, total_time, total_frame / total_time))
    return 


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)