import os
import numpy as np
import argparse
import json
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
from tqdm import tqdm

class UOTEvaluator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.success_thresholds = np.arange(0, 1.05, 0.05)  # 0 to 1 with step 0.05
        self.precision_thresholds = np.arange(0, 51, 1)    # 0 to 50 pixels
        self.norm_precision_thresholds = np.arange(0, 0.51, 0.01)  # 0 to 0.5 with step 0.01
        
        # Parameters for the new metrics
        self.csc_tau_c = 0.2  # Center threshold for CSC
        self.csc_tau_s = 0.2  # Scale threshold for CSC
        self.gas_sigma_c_factor = 0.5  # Factor of GT diagonal for GAS center tolerance
        self.gas_sigma_s_factor = 0.5  # Factor of GT w+h for GAS scale tolerance
        
    def compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes [x, y, w, h]."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1, y1, w1, h1 = bbox1[:4]
        x2, y2, w2, h2 = bbox2[:4]
        
        # Convert to [x1, y1, x2, y2] format
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Compute intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def compute_center_error(self, bbox1, bbox2):
        """Compute center location error between two bounding boxes."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return float('inf')
        
        x1, y1, w1, h1 = bbox1[:4]
        x2, y2, w2, h2 = bbox2[:4]
        
        center1 = [x1 + w1/2, y1 + h1/2]
        center2 = [x2 + w2/2, y2 + h2/2]
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def compute_normalized_center_error(self, bbox1, bbox2):
        """Compute normalized center location error."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return float('inf')
        
        center_error = self.compute_center_error(bbox1, bbox2)
        
        # Normalize by diagonal of ground truth bounding box
        _, _, w_gt, h_gt = bbox2[:4]
        diagonal = np.sqrt(w_gt**2 + h_gt**2)
        
        if diagonal <= 0:
            return float('inf')
        
        return center_error / diagonal

    def compute_gas(self, bbox1, bbox2):
        """Compute Geometric Alignment Score (GAS)."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1, y1, w1, h1 = bbox1[:4]
        x2, y2, w2, h2 = bbox2[:4]
        
        # Center term
        center_error_sq = (self.compute_center_error(bbox1, bbox2))**2
        diagonal = np.sqrt(w2**2 + h2**2)
        if diagonal <= 0:
            return 0.0
        
        sigma_c_sq = (self.gas_sigma_c_factor * diagonal)**2
        if sigma_c_sq == 0:
             return 0.0 # Avoid division by zero
        center_penalty = np.exp(-center_error_sq / sigma_c_sq)

        # Scale term
        scale_error_sq = (w1 - w2)**2 + (h1 - h2)**2
        sigma_s_sq = (self.gas_sigma_s_factor * (w2 + h2))**2
        if sigma_s_sq == 0:
             return 0.0 # Avoid division by zero
        scale_penalty = np.exp(-scale_error_sq / sigma_s_sq)

        return center_penalty * scale_penalty

    def compute_csc_frame(self, bbox1, bbox2):
        """Compute Center-Scale Consistency (CSC) for a single frame."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0

        # Center consistency
        norm_center_error = self.compute_normalized_center_error(bbox1, bbox2)
        center_ok = norm_center_error < self.csc_tau_c
        
        # Scale consistency
        w_p, h_p = bbox1[2], bbox1[3]
        w_g, h_g = bbox2[2], bbox2[3]

        if w_g <= 0 or h_g <= 0:
            return 0

        w_scale_ok = abs(w_p - w_g) / w_g < self.csc_tau_s
        h_scale_ok = abs(h_p - h_g) / h_g < self.csc_tau_s

        return 1 if center_ok and w_scale_ok and h_scale_ok else 0
    
    def compute_success_curve(self, ious):
        """Compute success curve (percentage of frames with IoU > threshold)."""
        success_curve = []
        for threshold in self.success_thresholds:
            success_rate = np.mean(np.array(ious) > threshold)
            success_curve.append(success_rate)
        return success_curve
    
    def compute_precision_curve(self, center_errors):
        """Compute precision curve (percentage of frames with center error < threshold)."""
        precision_curve = []
        valid_errors = [e for e in center_errors if e != float('inf')]
        
        for threshold in self.precision_thresholds:
            if len(valid_errors) == 0:
                precision_rate = 0.0
            else:
                precision_rate = np.mean(np.array(valid_errors) < threshold)
            precision_curve.append(precision_rate)
        return precision_curve
    
    def compute_normalized_precision_curve(self, norm_center_errors):
        """Compute normalized precision curve."""
        norm_precision_curve = []
        valid_errors = [e for e in norm_center_errors if e != float('inf')]
        
        for threshold in self.norm_precision_thresholds:
            if len(valid_errors) == 0:
                precision_rate = 0.0
            else:
                precision_rate = np.mean(np.array(valid_errors) < threshold)
            norm_precision_curve.append(precision_rate)
        return norm_precision_curve
    
    def evaluate_video(self, pred_bboxes, gt_bboxes, video_name):
        """Evaluate a single video sequence."""
        results = {
            'video_name': video_name,
            'num_frames': len(gt_bboxes),
            'valid_frames': 0,
            'ious': [],
            'center_errors': [],
            'norm_center_errors': [],
            'success_curve': [],
            'precision_curve': [],
            'norm_precision_curve': [],
            'auc_success': 0.0,
            'auc_precision': 0.0,
            'auc_norm_precision': 0.0,
            'success_score': 0.0,  # Success at IoU=0.5
            'precision_score': 0.0,  # Precision at 20 pixels
            'norm_precision_score': 0.0,  # Normalized precision at 0.2
            'gas_scores': [],
            'csc_frames': [], # 1s and 0s for each frame
            'mean_gas': 0.0,
            'csc_score': 0.0,
        }
        
        # Ensure same length
        min_frames = min(len(pred_bboxes), len(gt_bboxes))
        pred_bboxes = pred_bboxes[:min_frames]
        gt_bboxes = gt_bboxes[:min_frames]
        
        # Compute frame-wise metrics
        for i, (pred_bbox, gt_bbox) in enumerate(zip(pred_bboxes, gt_bboxes)):
            # Skip frames with invalid ground truth
            if len(gt_bbox) < 4 or all(x == 0 for x in gt_bbox[:4]):
                results['ious'].append(0.0)
                results['center_errors'].append(float('inf'))
                results['norm_center_errors'].append(float('inf'))
                results['gas_scores'].append(0.0)
                results['csc_frames'].append(0)
                continue
            
            # Skip frames with invalid predictions
            if len(pred_bbox) < 4 or all(x == 0 for x in pred_bbox[:4]):
                results['ious'].append(0.0)
                results['center_errors'].append(float('inf'))
                results['norm_center_errors'].append(float('inf'))
                results['gas_scores'].append(0.0)
                results['csc_frames'].append(0)
                continue
            
            results['valid_frames'] += 1
            
            # Compute IoU
            iou = self.compute_iou(pred_bbox, gt_bbox)
            results['ious'].append(iou)
            
            # Compute center error
            center_error = self.compute_center_error(pred_bbox, gt_bbox)
            results['center_errors'].append(center_error)
            
            # Compute normalized center error
            norm_center_error = self.compute_normalized_center_error(pred_bbox, gt_bbox)
            results['norm_center_errors'].append(norm_center_error)
            
            gas_score = self.compute_gas(pred_bbox, gt_bbox)
            results['gas_scores'].append(gas_score)
            
            csc_frame = self.compute_csc_frame(pred_bbox, gt_bbox)
            results['csc_frames'].append(csc_frame)
        
        # Compute curves
        results['success_curve'] = self.compute_success_curve(results['ious'])
        results['precision_curve'] = self.compute_precision_curve(results['center_errors'])
        results['norm_precision_curve'] = self.compute_normalized_precision_curve(results['norm_center_errors'])
        
        # Compute AUC scores
        results['auc_success'] = np.trapz(results['success_curve'], dx=0.05)
        results['auc_precision'] = np.trapz(results['precision_curve'], dx=1.0) / 50.0  # Normalize by range
        results['auc_norm_precision'] = np.trapz(results['norm_precision_curve'], dx=0.01) / 0.5  # Normalize by range
        
        # Compute specific threshold scores
        # Success at IoU = 0.5
        threshold_idx = np.where(self.success_thresholds >= 0.5)[0]
        if len(threshold_idx) > 0:
            results['success_score'] = results['success_curve'][threshold_idx[0]]
        
        # Precision at 20 pixels
        threshold_idx = np.where(self.precision_thresholds >= 20)[0]
        if len(threshold_idx) > 0:
            results['precision_score'] = results['precision_curve'][threshold_idx[0]]
        
        # Normalized precision at 0.2
        threshold_idx = np.where(self.norm_precision_thresholds >= 0.2)[0]
        if len(threshold_idx) > 0:
            results['norm_precision_score'] = results['norm_precision_curve'][threshold_idx[0]]

        valid_gas = [s for s in results['gas_scores'] if s > 0]
        if valid_gas:
            results['mean_gas'] = np.mean(valid_gas)

        if results['valid_frames'] > 0:
            results['csc_score'] = np.sum(results['csc_frames']) / results['valid_frames']
        else:
            results['csc_score'] = 0.0

        return results
    
    def load_ground_truth(self, dataset_path, video_name, dataset_name):
        """Load ground truth annotations for a video."""
        if dataset_name == "UOT32":
            gt_path = os.path.join(dataset_path, video_name, "groundtruth_rect.txt")
        elif dataset_name == "UTB180":
            gt_path = os.path.join(dataset_path, video_name, "groundtruth_rect.txt")
        elif dataset_name == "WebUOT-1M":
            gt_path = os.path.join(dataset_path, video_name, "groundtruth_rect.txt")
        elif dataset_name == "UWCOT220":
            gt_path = os.path.join(dataset_path, video_name, "groundtruth_rect.txt")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found: {gt_path}")
            return []
        
        gt_bboxes = []
        try:
            with open(gt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if ',' in line:
                            coords = list(map(float, line.split(',')))
                        else:
                            coords = list(map(float, line.split()))
                        
                        if len(coords) >= 4:
                            gt_bboxes.append(coords[:4])  # [x, y, w, h]
        except Exception as e:
            print(f"Error reading ground truth {gt_path}: {e}")
            return []
        
        return gt_bboxes
    
    def load_predictions(self, results_path, video_name):
        """Load predicted bounding boxes from result file."""
        pred_path = os.path.join(results_path, f"{video_name}.txt")
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file not found: {pred_path}")
            return []
        
        pred_bboxes = []
        try:
            with open(pred_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        if ',' in line:
                            coords = list(map(float, line.split(',')))
                        else:
                            coords = list(map(float, line.split()))
                        
                        if len(coords) >= 4:
                            pred_bboxes.append(coords[:4])  # [x, y, w, h]
        except Exception as e:
            print(f"Error reading predictions {pred_path}: {e}")
            return []
        
        return pred_bboxes
    
    def get_video_list(self, dataset_path, dataset_name):
        """Get list of videos for evaluation."""
        videos = []
        
        if dataset_name == "UOT32":
            video_dirs = [d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))]
            videos = sorted(video_dirs)
            
        elif dataset_name == "UTB180":
            utb180_base = os.path.join(dataset_path)
            if os.path.exists(utb180_base):
                video_dirs = [d for d in os.listdir(utb180_base) 
                              if os.path.isdir(os.path.join(utb180_base, d)) and d.startswith("Video_")]
                videos = sorted(video_dirs)
                
        elif dataset_name == "WebUOT-1M":
            test_frames_base = os.path.join(dataset_path)
            if os.path.exists(test_frames_base):
                video_dirs = [d for d in os.listdir(test_frames_base) 
                             if os.path.isdir(os.path.join(test_frames_base, d)) and d.startswith("WebUOT-1M_Test_")]
                videos = sorted(video_dirs)
                
        elif dataset_name == "UWCOT220":
            video_dirs = [d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))]
            videos = sorted(video_dirs)
        
        return videos
    
    def plot_curves(self, all_results, output_dir):
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        # ================================================================
        # 1. SUCCESS PLOT (Primary plot shown in papers)
        # ================================================================
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        success_curves = [result['success_curve'] for result in all_results.values()]
        mean_success = np.mean(success_curves, axis=0)
        auc_score = np.trapz(mean_success, dx=0.05)
        
        ax.plot(self.success_thresholds, mean_success, 'b-', linewidth=3, 
                label=f'Our Method [{auc_score:.3f}]')
        
        # Add confidence interval (optional)
        std_success = np.std(success_curves, axis=0)
        ax.fill_between(self.success_thresholds, 
                       np.maximum(0, mean_success - std_success), 
                       np.minimum(1, mean_success + std_success), 
                       alpha=0.2, color='blue')
        
        ax.set_xlabel('Overlap Threshold')
        ax.set_ylabel('Success Rate')
        ax.set_title(f'Success Plot on {self.dataset_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add IoU=0.5 reference line (commonly used threshold)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
        ax.text(0.52, 0.1, 'IoU=0.5', rotation=90, verticalalignment='bottom', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.dataset_name}_success_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{self.dataset_name}_success_plot.pdf'), 
                   bbox_inches='tight')
        plt.close()
        
        # ================================================================
        # 2. PRECISION PLOT (Secondary plot shown in papers)
        # ================================================================
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        precision_curves = [result['precision_curve'] for result in all_results.values()]
        mean_precision = np.mean(precision_curves, axis=0)
        prec_20 = mean_precision[20] if len(mean_precision) > 20 else 0
        
        ax.plot(self.precision_thresholds, mean_precision, 'r-', linewidth=3, 
                label=f'Our Method [{prec_20:.3f}]')
        
        # Add confidence interval
        std_precision = np.std(precision_curves, axis=0)
        ax.fill_between(self.precision_thresholds, 
                       np.maximum(0, mean_precision - std_precision), 
                       np.minimum(1, mean_precision + std_precision), 
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Location Error Threshold (pixels)')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision Plot on {self.dataset_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1)
        
        # Add 20px reference line (standard reporting threshold)
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.7)
        ax.text(21, 0.1, '20px', rotation=90, verticalalignment='bottom', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.dataset_name}_precision_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{self.dataset_name}_precision_plot.pdf'), 
                   bbox_inches='tight')
        plt.close()
        
        # ================================================================
        # 3. COMBINED SUCCESS & PRECISION PLOT (Often shown side-by-side)
        # ================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Success plot
        ax1.plot(self.success_thresholds, mean_success, 'b-', linewidth=3, 
                label=f'Our Method [{auc_score:.3f}]')
        ax1.fill_between(self.success_thresholds, 
                        np.maximum(0, mean_success - std_success), 
                        np.minimum(1, mean_success + std_success), 
                        alpha=0.2, color='blue')
        ax1.set_xlabel('Overlap Threshold')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Plot')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Precision plot
        ax2.plot(self.precision_thresholds, mean_precision, 'r-', linewidth=3, 
                label=f'Our Method [{prec_20:.3f}]')
        ax2.fill_between(self.precision_thresholds, 
                        np.maximum(0, mean_precision - std_precision), 
                        np.minimum(1, mean_precision + std_precision), 
                        alpha=0.2, color='red')
        ax2.set_xlabel('Location Error Threshold (pixels)')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision Plot')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0, 50)
        ax2.set_ylim(0, 1)
        ax2.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Tracking Performance on {self.dataset_name}', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.dataset_name}_combined_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{self.dataset_name}_combined_plots.pdf'), 
                   bbox_inches='tight')
        plt.close()
        
        # ================================================================
        # 4. IoU DISTRIBUTION & STATISTICS
        # ================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # IoU histogram
        all_ious = []
        for result in all_results.values():
            all_ious.extend([iou for iou in result['ious'] if iou > 0])
        
        if all_ious:
            ax1.hist(all_ious, bins=50, alpha=0.7, color='skyblue', density=True, edgecolor='black')
            ax1.axvline(x=np.mean(all_ious), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(all_ious):.3f}')
            ax1.axvline(x=np.median(all_ious), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(all_ious):.3f}')
            ax1.axvline(x=0.5, color='orange', linestyle='--', linewidth=2,
                       label='Success Threshold (0.5)')
            ax1.set_xlabel('IoU')
            ax1.set_ylabel('Density')
            ax1.set_title('IoU Distribution')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_xlim(0, 1)
        
        # Box plot of IoU per video
        video_ious = []
        video_names = []
        for video_name, result in all_results.items():
            valid_ious = [iou for iou in result['ious'] if iou > 0]
            if valid_ious:
                video_ious.append(valid_ious)
                video_names.append(video_name[:10] + '...' if len(video_name) > 10 else video_name)
        
        if video_ious:
            ax2.boxplot(video_ious, labels=video_names)
            ax2.set_ylabel('IoU')
            ax2.set_title('Per-Video IoU Distribution')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Limit number of videos shown for readability
            if len(video_names) > 10:
                step = len(video_names) // 10
                ax2.set_xticks(range(0, len(video_names), step))
                ax2.set_xticklabels([video_names[i] for i in range(0, len(video_names), step)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.dataset_name}_iou_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # ================================================================
        # 5. PERFORMANCE OVER TIME ANALYSIS
        # ================================================================
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Analyze performance degradation over sequence length
        max_frames = max([len(result['ious']) for result in all_results.values()])
        frame_bins = np.arange(0, min(max_frames, 200), 10)  # Limit to 200 frames for clarity
        
        mean_ious_over_time = []
        std_ious_over_time = []
        
        for frame_idx in frame_bins:
            frame_ious = []
            for result in all_results.values():
                if frame_idx < len(result['ious']) and result['ious'][frame_idx] > 0:
                    frame_ious.append(result['ious'][frame_idx])
            
            if frame_ious:
                mean_ious_over_time.append(np.mean(frame_ious))
                std_ious_over_time.append(np.std(frame_ious))
            else:
                mean_ious_over_time.append(0)
                std_ious_over_time.append(0)
        
        mean_ious_over_time = np.array(mean_ious_over_time)
        std_ious_over_time = np.array(std_ious_over_time)
        
        ax.plot(frame_bins, mean_ious_over_time, 'b-', linewidth=2, label='Mean IoU')
        ax.fill_between(frame_bins, 
                       np.maximum(0, mean_ious_over_time - std_ious_over_time),
                       np.minimum(1, mean_ious_over_time + std_ious_over_time),
                       alpha=0.3, color='blue', label='±1 std')
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Success Threshold')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('IoU')
        ax.set_title(f'Tracking Performance Over Time - {self.dataset_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.dataset_name}_temporal_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # ================================================================
        # 6. NEW GEOMETRIC METRICS ANALYSIS
        # ================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # GAS Distribution
        all_gas = []
        for result in all_results.values():
            all_gas.extend([s for s in result['gas_scores'] if s > 0])
            
        if all_gas:
            ax2.hist(all_gas, bins=50, alpha=0.7, color='green', density=True, edgecolor='black')
            mean_gas = np.mean(all_gas)
            ax2.axvline(x=mean_gas, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_gas:.3f}')
            ax2.set_xlabel('Geometric Alignment Score (GAS)')
            ax2.set_ylabel('Density')
            ax2.set_title('GAS Distribution (Higher is Better)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_xlim(0, 1)

        plt.suptitle(f'Geometric Alignment Metrics on {self.dataset_name}', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.dataset_name}_geometric_metrics_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generated evaluation plots:")
        print(f"  - Success plot: {self.dataset_name}_success_plot.png/pdf")
        print(f"  - Precision plot: {self.dataset_name}_precision_plot.png/pdf") 
        print(f"  - Combined plots: {self.dataset_name}_combined_plots.png/pdf")
        print(f"  - IoU analysis: {self.dataset_name}_iou_analysis.png")
        print(f"  - Temporal analysis: {self.dataset_name}_temporal_analysis.png")
        print(f"  - Geometric metrics analysis: {self.dataset_name}_geometric_metrics_analysis.png")
    
    def evaluate_dataset(self, dataset_path, results_path, output_dir=None):
        """Evaluate entire dataset."""
        if output_dir is None:
            output_dir = os.path.join(results_path, 'evaluation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        videos = self.get_video_list(dataset_path, self.dataset_name)
        
        if not videos:
            print(f"No videos found for dataset {self.dataset_name}")
            return None
        
        print(f"Evaluating {len(videos)} videos from {self.dataset_name} dataset...")
        
        all_results = {}
        summary_stats = {
            'dataset': self.dataset_name,
            'num_videos': len(videos),
            'total_frames': 0,
            'valid_frames': 0,
            'mean_auc_success': 0.0,
            'mean_auc_precision': 0.0,
            'mean_auc_norm_precision': 0.0,
            'mean_success_score': 0.0,
            'mean_precision_score': 0.0,
            'mean_norm_precision_score': 0.0,
            'mean_iou': 0.0,
            'std_iou': 0.0,
            'mean_gas': 0.0,
            'mean_csc_score': 0.0,
            'video_results': []
        }
        
        # blacklist = ["FishFollowing"]
        for video in tqdm(videos, desc=f"Evaluating {self.dataset_name}"):
            # if video not in blacklist:
            #     continue
            # Load ground truth
            gt_bboxes = self.load_ground_truth(dataset_path, video, self.dataset_name)
            
            # Load predictions
            pred_bboxes = self.load_predictions(results_path, video)
            
            if not gt_bboxes or not pred_bboxes:
                print(f"Skipping {video}: missing ground truth or predictions")
                continue
            
            # Evaluate video
            result = self.evaluate_video(pred_bboxes, gt_bboxes, video)
            all_results[video] = result
            
            # Add to summary
            summary_stats['total_frames'] += result['num_frames']
            summary_stats['valid_frames'] += result['valid_frames']
            
            video_summary = {
                'video_name': video,
                'num_frames': result['num_frames'],
                'valid_frames': result['valid_frames'],
                'auc_success': result['auc_success'],
                'auc_precision': result['auc_precision'],
                'auc_norm_precision': result['auc_norm_precision'],
                'success_score': result['success_score'],
                'precision_score': result['precision_score'],
                'norm_precision_score': result['norm_precision_score'],
                'mean_iou': np.mean([iou for iou in result['ious'] if iou > 0]) if result['ious'] else 0.0,
                'mean_gas': result['mean_gas'],
                'csc_score': result['csc_score'],
            }
            summary_stats['video_results'].append(video_summary)
        
        if not all_results:
            print("No valid results found!")
            return None
        
        # Compute overall statistics
        auc_success_scores = [r['auc_success'] for r in all_results.values()]
        auc_precision_scores = [r['auc_precision'] for r in all_results.values()]
        auc_norm_precision_scores = [r['auc_norm_precision'] for r in all_results.values()]
        success_scores = [r['success_score'] for r in all_results.values()]
        precision_scores = [r['precision_score'] for r in all_results.values()]
        norm_precision_scores = [r['norm_precision_score'] for r in all_results.values()]
        
        all_ious = []
        for result in all_results.values():
            all_ious.extend([iou for iou in result['ious'] if iou > 0])
        
        mean_gas_scores = [r['mean_gas'] for r in all_results.values()]
        csc_scores = [r['csc_score'] for r in all_results.values()]

        summary_stats['mean_auc_success'] = np.mean(auc_success_scores)
        summary_stats['mean_auc_precision'] = np.mean(auc_precision_scores)
        summary_stats['mean_auc_norm_precision'] = np.mean(auc_norm_precision_scores)
        summary_stats['mean_success_score'] = np.mean(success_scores)
        summary_stats['mean_precision_score'] = np.mean(precision_scores)
        summary_stats['mean_norm_precision_score'] = np.mean(norm_precision_scores)
        summary_stats['mean_iou'] = np.mean(all_ious) if all_ious else 0.0
        summary_stats['std_iou'] = np.std(all_ious) if all_ious else 0.0
        summary_stats['mean_gas'] = np.mean(mean_gas_scores) if mean_gas_scores else 0.0
        summary_stats['mean_csc_score'] = np.mean(csc_scores) if csc_scores else 0.0
        
        # Save results
        results_file = os.path.join(output_dir, f'{self.dataset_name}_detailed_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for video, result in all_results.items():
                json_result = result.copy()
                for key in ['success_curve', 'precision_curve', 'norm_precision_curve', 
                           'ious', 'center_errors', 'norm_center_errors',
                           'gas_scores', 'csc_frames']:
                    if isinstance(json_result[key], np.ndarray):
                        json_result[key] = json_result[key].tolist()
                    elif isinstance(json_result[key], list):
                        json_result[key] = [x if x != float('inf') else None for x in json_result[key]]
                json_results[video] = json_result
            
            json.dump(json_results, f, indent=2)
        
        summary_file = os.path.join(output_dir, f'{self.dataset_name}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Generate plots
        self.plot_curves(all_results, output_dir)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {self.dataset_name}")
        print(f"{'='*60}")
        print(f"Videos evaluated: {len(all_results)}")
        print(f"Total frames: {summary_stats['total_frames']}")
        print(f"Valid frames: {summary_stats['valid_frames']}")
        print(f"\nSUCCESS METRICS:")
        print(f"  AUC Success: {summary_stats['mean_auc_success']:.4f}")
        print(f"  Success@0.5: {summary_stats['mean_success_score']:.4f}")
        print(f"\nPRECISION METRICS:")
        print(f"  AUC Precision: {summary_stats['mean_auc_precision']:.4f}")
        print(f"  Precision@20px: {summary_stats['mean_precision_score']:.4f}")
        print(f"\nIOU METRICS:")
        print(f"  Mean IoU: {summary_stats['mean_iou']:.4f} ± {summary_stats['std_iou']:.4f}")
        print(f"\nGEOMETRIC ALIGNMENT METRICS:")
        print(f"  Mean GAS (higher is better): {summary_stats['mean_gas']:.4f}")
        print(f"  Mean CSC@0.2 (higher is better): {summary_stats['mean_csc_score']:.4f}")
        print(f"\nResults saved to: {output_dir}")
        print(f"{'='*60}")
        
        return summary_stats

def main():
    parser = argparse.ArgumentParser(description='Unified UOT Evaluation Script')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['UOT32', 'UTB180', 'WebUOT-1M', 'UWCOT220'],
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to folder containing prediction .txt files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for evaluation results (default: results_path/evaluation_results)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return
    
    if not os.path.exists(args.results_path):
        print(f"Error: Results path does not exist: {args.results_path}")
        return
    
    # Create evaluator
    evaluator = UOTEvaluator(args.dataset)
    
    # Run evaluation
    evaluator.evaluate_dataset(
        args.dataset_path, 
        args.results_path,
        args.output_dir
    )

if __name__ == '__main__':
    main()