import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import json
import argparse


class BBoxSeqDataset(Dataset):
    def __init__(self, annotation_file, img_root, transform=None, num_views=2):
        with open(annotation_file, "r") as f:
            self.data = json.load(f)

        self.samples = []
        for seq in self.data["sequences"]:
            frames = seq["frames"]

            if isinstance(frames, dict):
                frames = list(frames.values())

            # Group by track_id, keep frames with bbox
            tracks = {}
            for f in frames:
                if "track_id" not in f or f.get("bbox") is None:
                    continue
                tid = f["track_id"]
                tracks.setdefault(tid, []).append(f)

            for tid, track_frames in tracks.items():
                if len(track_frames) > 1:
                    self.samples.append((seq["sequence_dir"], track_frames))

        self.img_root = img_root
        self.transform = transform
        self.num_views = num_views

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, frames = self.samples[idx]

        # Select anchor frame and random positive frame
        anchor_frame = random.choice(frames)
        positive_frame = random.choice(frames)

        # Load and crop images
        img_anchor = Image.open(f"{self.img_root}/{seq_dir}/{anchor_frame['file_name']}").convert("RGB")
        img_positive = Image.open(f"{self.img_root}/{seq_dir}/{positive_frame['file_name']}").convert("RGB")

        if anchor_frame.get("bbox") is not None:
            x, y, w, h = anchor_frame["bbox"]
            img_anchor = img_anchor.crop((x, y, x + w, y + h))
        if positive_frame.get("bbox") is not None:
            x, y, w, h = positive_frame["bbox"]
            img_positive = img_positive.crop((x, y, x + w, y + h))

        if self.transform:
            img_anchor = self.transform(img_anchor)
            img_positive = self.transform(img_positive)

        def beer_lambert_simple(img):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            arr = np.array(img).astype(np.float32) / 255.0
            h, w, c = arr.shape
            depth = np.linspace(0, 1, h, dtype=np.float32)[:, None]
            beta = random.uniform(0.1, 0.5)  # Random attenuation coefficient
            transmission = np.exp(-beta * depth)[:, :, None]
            # Apply blue-green underwater tint (average values estimated from SOT data)
            arr = arr * transmission + (1 - transmission) * np.array([0.6, 0.8, 0.9])[None, None, :]
            return Image.fromarray(np.uint8(np.clip(arr * 255, 0, 255)))

        img_physics = beer_lambert_simple(img_anchor)
        if self.transform:
            img_physics = self.transform(img_physics)

        return [img_anchor, img_positive, img_physics], idx

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128, pretrained=True):
        super().__init__()
        if base_model == "resnet18":
            base = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif base_model == "resnet50":
            base = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            raise NotImplementedError(f"Model {base_model} not implemented")

        self.encoder = nn.Sequential(*list(base.children())[:-1])
        dim_mlp = base.fc.in_features
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze()
        if len(h.shape) == 1:
            h = h.unsqueeze(0)
        z = self.projector(h)
        return z

def contrastive_loss(features, margin=0.5, temperature=0.1, alpha=1.0):
    """Improved contrastive loss with temperature scaling."""
    batch_size = features.shape[0] // 2
    
    z1 = features[:batch_size]
    z2 = features[batch_size:]
    
    # Positive loss
    pos_sim = F.cosine_similarity(z1, z2, dim=1) / temperature
    pos_loss = -torch.log(torch.sigmoid(pos_sim)).mean()
    
    # Negative loss using pairwise similarities
    all_z1_z1 = torch.matmul(z1, z1.T) / temperature
    all_z2_z2 = torch.matmul(z2, z2.T) / temperature
    all_z1_z2 = torch.matmul(z1, z2.T) / temperature
    all_z2_z1 = torch.matmul(z2, z1.T) / temperature
    
    mask = torch.eye(batch_size, device=features.device).bool()
    
    neg_z1_z1 = all_z1_z1[~mask]
    neg_z2_z2 = all_z2_z2[~mask]
    neg_z1_z2 = all_z1_z2[~mask]
    neg_z2_z1 = all_z2_z1[~mask]
    
    all_neg_sims = torch.cat([neg_z1_z1, neg_z2_z2, neg_z1_z2, neg_z2_z1])
    neg_loss = -torch.log(torch.sigmoid(-all_neg_sims + margin)).mean()
    
    total_loss = pos_loss + alpha * neg_loss
    
    return total_loss, pos_loss, neg_loss


def triplet_contrastive_loss(features, margin=0.5, hard_mining=True):
    """Triplet-based contrastive loss with hard negative mining."""
    batch_size = features.shape[0] // 2
    
    z1 = features[:batch_size]
    z2 = features[batch_size:]
    
    losses = []
    
    for i in range(batch_size):
        anchor = z1[i]
        positive = z2[i]
        
        pos_dist = 1 - F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
        
        neg_candidates = []
        for j in range(batch_size):
            if i != j:
                neg_dist_1 = 1 - F.cosine_similarity(anchor.unsqueeze(0), z1[j].unsqueeze(0))
                neg_dist_2 = 1 - F.cosine_similarity(anchor.unsqueeze(0), z2[j].unsqueeze(0))
                neg_candidates.extend([neg_dist_1, neg_dist_2])
        
        if len(neg_candidates) > 0:
            neg_candidates = torch.stack(neg_candidates)
            
            if hard_mining:
                neg_dist = neg_candidates.min()
            else:
                neg_dist = neg_candidates.mean()
            
            triplet_loss = torch.clamp(pos_dist - neg_dist + margin, min=0)
            losses.append(triplet_loss)
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=features.device)


def nt_xent_loss(features, temperature=0.07):
    """NT-Xent loss for SimCLR."""
    batch_size = features.shape[0] // 2
    
    sim = torch.matmul(features, features.T) / temperature
    
    # Numerical stability
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    
    # Create masks
    pos_mask = torch.zeros(2 * batch_size, 2 * batch_size).to(features.device)
    for i in range(batch_size):
        pos_mask[i, i + batch_size] = 1
        pos_mask[i + batch_size, i] = 1
    
    neg_mask = torch.ones(2 * batch_size, 2 * batch_size).to(features.device)
    neg_mask = neg_mask - torch.eye(2 * batch_size).to(features.device) - pos_mask
    
    # Compute loss
    exp_sim = torch.exp(sim)
    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    neg_sim = (exp_sim * neg_mask).sum(dim=1)
    
    loss = -torch.log(pos_sim / (pos_sim + neg_sim))
    
    return loss.mean()


def visualize_embeddings(model, dataloader, device, epoch, out_dir, max_batches=1):
    """Create t-SNE visualization of embeddings."""
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for i, (views, indices) in enumerate(dataloader):
            if i >= max_batches:
                break
                
            x = views[0].to(device)
            feats = model.encoder(x).view(x.size(0), -1)
            all_feats.append(feats.cpu())
            all_labels.extend(indices.tolist())

    if len(all_feats) == 0:
        return
        
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_labels = np.array(all_labels)

    tsne = TSNE(n_components=2, perplexity=min(30, len(all_feats)-1), random_state=42)
    feats_2d = tsne.fit_transform(all_feats)

    np.save(f"{out_dir}/epoch{epoch:03d}_embeddings.npy", all_feats)

    plt.figure(figsize=(8, 6))
    plt.scatter(feats_2d[:,0], feats_2d[:,1], c=all_labels, cmap='tab20', s=20, alpha=0.7)
    plt.title(f"t-SNE of embeddings (epoch {epoch})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/epoch{epoch:03d}_tsne.png", dpi=150)
    plt.close()


def train(annotation_file, img_root, epochs=50, batch_size=16, num_views=2, lr=1e-3, 
          temperature=0.07, out_dir="./outputs", save_freq=1, loss_type="improved_contrastive", resume_path=None):
    """Main training function."""
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    dataset = BBoxSeqDataset(
        annotation_file=annotation_file,
        img_root=img_root,
        transform=transform,
        num_views=num_views
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       drop_last=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using loss type: {loss_type}")
    print(f"Dataset size: {len(dataset)}")
    
    model = ResNetSimCLR(pretrained=True).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5}, 
        {"params": model.projector.parameters(), "lr": 1e-3}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}.")

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for views, _ in pbar:
            views = [v.to(device) for v in views]
            all_views = torch.cat(views, dim=0)
            
            features = model(all_views)
            
            if loss_type == "contrastive":
                loss = contrastive_loss(features, margin=1.0, temperature=temperature)
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                
            elif loss_type == "improved_contrastive":
                loss = nt_xent_loss(features, temperature=temperature)
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                
            elif loss_type == "triplet":
                loss = triplet_contrastive_loss(features, margin=0.5, hard_mining=True)
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                
            else:
                loss = nt_xent_loss(features, temperature)
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, f"checkpoint_epoch_{epoch+1}.pth"))

        # Visualization
        if (epoch + 1) % save_freq == 0:
            visualize_embeddings(model, loader, device, epoch+1, out_dir)

    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pth"))
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-Informed Contrastive Learning")
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--img_root', type=str, required=True, help='Path to image root directory')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for contrastive loss')
    parser.add_argument('--out_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--loss_type', type=str, default='improved_contrastive', 
                       choices=['contrastive', 'improved_contrastive', 'triplet'], 
                       help='Type of contrastive loss')
    
    args = parser.parse_args()

    train(
        annotation_file=args.annotation_file,
        img_root=args.img_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_views=2,
        lr=args.lr,
        temperature=args.temperature,
        out_dir=args.out_dir,
        loss_type=args.loss_type,
        resume_path=args.resume_checkpoint
    )