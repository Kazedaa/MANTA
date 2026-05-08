import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from collections import defaultdict
import os

import warnings
warnings.filterwarnings("ignore")

class PhysicsResNet18(nn.Module):
    def __init__(self, pretrained=False, out_dim=128):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # drop FC

    def forward(self, x):
        feat = self.encoder(x)
        return feat.view(feat.size(0), -1)


def get_physics_resnet18_embedder(checkpoint_path='model_epoch_1.pth', device="cuda"):
    # initialize encoder (structure must match training)
    model = PhysicsResNet18(pretrained=False).to(device)

    # load checkpoint (we saved {"model": model.state_dict(), "optimizer": ...})
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # remove projection head keys if they exist
    filtered_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if "encoder." in k}

    # load into encoder
    model.encoder.load_state_dict(filtered_state, strict=False)
    model.eval()

    class Embedder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def forward(self, x):
            feat = self.backbone(x)
            return feat.view(feat.size(0), -1)

    return Embedder(model.encoder).to(device).eval()


# ---------- Step 1. Setup embedding model ----------
def get_resnet18_embedder(device="cuda"):
    resnet = models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop FC

    class Embedder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def forward(self, x):
            feat = self.backbone(x)
            return feat.view(feat.size(0), -1)
    
    return Embedder(backbone).to(device).eval()


# ---------- Step 2. Tracklet descriptor ----------
def get_tracklet_descriptor(tracklet, model, frame_folder, num_samples=10, sim_thresh=0.7, device="cuda"):
    img_list = sorted(os.listdir(f"{frame_folder}"), key=lambda x: int(x.split('.')[0]))
    frames = sorted([f for f, _ in tracklet])
    if len(frames) <= num_samples:
        sampled_frames = frames
    else:
        sampled_frames = np.linspace(frames[0], frames[-1], num_samples, dtype=int)

    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    embeddings = []
    with torch.no_grad():
        for f in sampled_frames:
            # find matching row
            boxes = [b for fid,b in tracklet if fid == f]
            if not boxes: 
                continue
            x1,y1,x2,y2 = map(int, boxes[0])
            img_path = f"{frame_folder}/{img_list[int(f)]}"
            img = Image.open(img_path).convert("RGB")
            crop = img.crop((x1, y1, x2, y2))
            inp = preprocess(crop).unsqueeze(0).to(device)
            emb = model(inp).cpu().numpy().squeeze()
            embeddings.append(emb)

    if len(embeddings) == 0:
        return None

    embeddings = np.stack(embeddings)
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_matrix = np.dot(normed, normed.T)

    avg_sims = sim_matrix.mean(axis=1)
    keep = avg_sims >= sim_thresh
    if not np.any(keep):
        keep[:] = True

    filtered_embeddings = embeddings[keep]
    descriptor = filtered_embeddings.mean(axis=0)
    descriptor /= np.linalg.norm(descriptor) + 1e-8
    return descriptor


# ---------- Step 3. Compare two tracklets ----------
def are_tracks_similar(tracklet_boxes, trackletA, trackletB, model, frame_folder, threshold=0.7, device="cuda"):
    if trackletA == trackletB:
        return True, 1.0
    trackA = tracklet_boxes[trackletA]
    trackB = tracklet_boxes[trackletB]

    descA = get_tracklet_descriptor(trackA, model, frame_folder, device=device)
    descB = get_tracklet_descriptor(trackB, model, frame_folder, device=device)
    if descA is None or descB is None:
        return False, 0.0
    sim = np.dot(descA, descB)
    return sim >= threshold, sim

import matplotlib.pyplot as plt
import random

def save_tracklet_samples(trackletA, trackletB, tracklet_boxes, frame_folder, sim_score=None, n_samples=2, save_path="tracklet_compare.jpg"):
    """
    Save n_samples random frames from two tracklets side by side as an image file.
    """
    def load_samples(tracklet, n):
        frames = [fid for fid,_ in tracklet]
        chosen = random.sample(frames, min(len(frames), n))
        samples = []
        for f in chosen:
            boxes = [b for fid,b in tracklet if fid == f]
            if not boxes: 
                continue
            x1,y1,x2,y2 = map(int, boxes[0])
            img_path = f"{frame_folder}/{int(f+1)}.jpg"   # adjust if frames are 0-indexed
            img = Image.open(img_path).convert("RGB")
            plt_img = np.array(img)
            # draw bbox
            plt_img = plt_img.copy()
            plt_img[y1:y1+2, x1:x2] = [255,0,0]
            plt_img[y2-2:y2, x1:x2] = [255,0,0]
            plt_img[y1:y2, x1:x1+2] = [255,0,0]
            plt_img[y1:y2, x2-2:x2] = [255,0,0]
            samples.append(plt_img)
        return samples

    A_samples = load_samples(tracklet_boxes[trackletA], n_samples)
    B_samples = load_samples(tracklet_boxes[trackletB], n_samples)

    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 6))
    if sim_score is not None:
        fig.suptitle(f"Track {trackletA} vs Track {trackletB} | CosSim={sim_score:.4f}", fontsize=14)

    for i, img in enumerate(A_samples):
        axes[0, i].imshow(img)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Track {trackletA}")
    for i, img in enumerate(B_samples):
        axes[1, i].imshow(img)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Track {trackletB}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")

# ---------- Step 4. Example usage ----------
if __name__ == "__main__":
    csv_file = "/mnt/50f2d3fa-182a-44fa-9f1f-0cabddd2c5a7/Harddisk/Manogna/suhas/manta/GOAT_Tracker/dumps/dumps_OCSort/FishFollowing.txt"
    frame_folder = "/mnt/50f2d3fa-182a-44fa-9f1f-0cabddd2c5a7/Harddisk/Manogna/suhas/manta/UOT32/FishFollowing/img"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = get_resnet18_embedder(device)
    model = get_physics_resnet18_embedder(checkpoint_path="model_epoch_1.pth", device=device)

    # Load CSV
    df = pd.read_csv(csv_file, header=None)
    df = df[[0,1,6,7,8,9]]
    df.columns = ["frame_id","track_id","x1","y1","x2","y2"]

    tracklet_boxes = defaultdict(list)
    for _, row in df.iterrows():
        tracklet_boxes[int(row.track_id)].append((row.frame_id, [row.x1,row.y1,row.x2,row.y2]))

    tA, tB = 774, 1995
    print(f"Num Frames A: {len(tracklet_boxes[tA])}, Num Frames B: {len(tracklet_boxes[tB])}")

    similar, score = are_tracks_similar(tracklet_boxes, tA, tB, model, frame_folder, threshold=0.9, device=device)
    print(f"Track {tA} vs {tB}: Similar? {similar}, Cosine similarity={score:.4f}")

    save_tracklet_samples(tA, tB, tracklet_boxes, frame_folder, sim_score=score, n_samples=2, save_path="tracklet_comparison.jpg")

