"""
VeRi-776 기반 CLIP fine-tuning for Vehicle Re-ID
Triplet Loss + ID Classification Loss

사용법:
  python3 train_reid.py --data /tmp/veri/VeRi --epochs 30 --batch 64
"""

import argparse
import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import open_clip


# ============================================================
# 1. VeRi Dataset
# ============================================================
class VeRiDataset(Dataset):
    """VeRi-776 데이터셋. 파일명: {vID}_c{camID}_{timestamp}_{binary}.jpg"""

    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []  # (path, vehicle_id, camera_id)
        self.vid_to_idx = {}  # vehicle_id -> label index
        self.vid_to_samples = defaultdict(list)  # for triplet mining

        exts = ('.jpg', '.jpeg', '.png')
        for f in sorted(self.img_dir.iterdir()):
            if f.suffix.lower() not in exts:
                continue
            parts = f.stem.split('_')
            vid = int(parts[0])
            cam = int(parts[1].replace('c', ''))

            if vid not in self.vid_to_idx:
                self.vid_to_idx[vid] = len(self.vid_to_idx)

            idx = len(self.samples)
            self.samples.append((str(f), vid, cam))
            self.vid_to_samples[vid].append(idx)

        self.num_classes = len(self.vid_to_idx)
        print(f"[Dataset] {img_dir}: {len(self.samples)}장, {self.num_classes}대 차량")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, vid, cam = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.vid_to_idx[vid]
        return img, label, vid


class TripletBatchSampler:
    """P x K 배치: P개 차량, 각 K장씩 = batch_size"""

    def __init__(self, dataset, p=16, k=4):
        self.p = p
        self.k = k
        self.dataset = dataset
        # K장 이상 있는 차량만 사용
        self.valid_vids = [vid for vid, idxs in dataset.vid_to_samples.items()
                           if len(idxs) >= k]
        print(f"[Sampler] P={p}, K={k}, 유효 차량: {len(self.valid_vids)}/{dataset.num_classes}")

    def __iter__(self):
        random.shuffle(self.valid_vids)
        for start in range(0, len(self.valid_vids) - self.p + 1, self.p):
            batch_vids = self.valid_vids[start:start + self.p]
            batch = []
            for vid in batch_vids:
                idxs = self.dataset.vid_to_samples[vid]
                chosen = random.choices(idxs, k=self.k) if len(idxs) < self.k else random.sample(idxs, self.k)
                batch.extend(chosen)
            yield batch

    def __len__(self):
        return len(self.valid_vids) // self.p


# ============================================================
# 2. Model: CLIP visual encoder + projection head
# ============================================================
class VehicleReIDModel(nn.Module):
    def __init__(self, clip_model, embed_dim=512, num_classes=776):
        super().__init__()
        self.clip_visual = clip_model.visual
        clip_out_dim = clip_model.visual.output_dim  # 512 for ViT-B-32

        # Projection head for Re-ID
        self.proj = nn.Sequential(
            nn.Linear(clip_out_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # ID classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # CLIP visual features (frozen or fine-tuned)
        feat = self.clip_visual(x)  # (B, clip_out_dim)
        feat = feat.float()  # fp16 -> fp32

        # Projection
        embed = self.proj(feat)  # (B, embed_dim)
        embed_normed = F.normalize(embed, p=2, dim=1)

        # Classification logits
        logits = self.classifier(embed)

        return embed_normed, logits


# ============================================================
# 3. Losses
# ============================================================
class TripletLoss(nn.Module):
    """Batch-hard triplet loss"""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        dist_mat = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        labels = labels.unsqueeze(0)
        same_id = labels == labels.t()

        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0

        for i in range(embeddings.size(0)):
            # Hardest positive
            pos_mask = same_id[i].clone()
            pos_mask[i] = False
            if not pos_mask.any():
                continue
            hardest_pos = dist_mat[i][pos_mask].max()

            # Hardest negative
            neg_mask = ~same_id[i]
            if not neg_mask.any():
                continue
            hardest_neg = dist_mat[i][neg_mask].min()

            loss += F.relu(hardest_pos - hardest_neg + self.margin)
            count += 1

        return loss / max(count, 1)


# ============================================================
# 4. Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, query_loader, gallery_loader, device):
    model.eval()

    def extract_all(loader):
        feats, vids, cams = [], [], []
        for imgs, labels, vehicle_ids in loader:
            imgs = imgs.to(device)
            embed, _ = model(imgs)
            feats.append(embed.cpu())
            vids.extend(vehicle_ids.numpy())
        return torch.cat(feats), np.array(vids)

    q_feats, q_vids = extract_all(query_loader)
    g_feats, g_vids = extract_all(gallery_loader)

    # Cosine similarity
    sim = q_feats @ g_feats.t()  # (num_q, num_g)

    # CMC & mAP
    cmc = np.zeros(10)
    ap_sum = 0
    num_valid = 0

    for i in range(len(q_vids)):
        scores = sim[i].numpy()
        order = np.argsort(-scores)
        matches = (g_vids[order] == q_vids[i]).astype(np.float32)

        if matches.sum() == 0:
            continue

        num_valid += 1
        # CMC
        first_match = np.where(matches > 0)[0][0]
        for k in range(10):
            if first_match < k + 1:
                cmc[k] += 1

        # AP
        num_rel = matches.sum()
        cum_match = np.cumsum(matches)
        precision = cum_match / (np.arange(len(matches)) + 1)
        ap_sum += (precision * matches).sum() / num_rel

    cmc /= max(num_valid, 1)
    mAP = ap_sum / max(num_valid, 1)

    return mAP, cmc[0], cmc[4]


# ============================================================
# 5. Training
# ============================================================
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    # CLIP 모델 로드
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu'
    )

    # 데이터셋
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    data_root = Path(args.data)
    train_ds = VeRiDataset(data_root / 'image_train', train_transform)
    query_ds = VeRiDataset(data_root / 'image_query', eval_transform)
    test_ds = VeRiDataset(data_root / 'image_test', eval_transform)

    sampler = TripletBatchSampler(train_ds, p=args.p, k=args.k)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_ds, batch_size=128, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # 모델
    model = VehicleReIDModel(clip_model, embed_dim=512, num_classes=train_ds.num_classes)
    model = model.to(device)

    # CLIP visual encoder: 낮은 lr로 fine-tune
    param_groups = [
        {'params': model.clip_visual.parameters(), 'lr': args.lr * 0.1},
        {'params': model.proj.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    triplet_loss = TripletLoss(margin=0.3)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 학습
    best_mAP = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_tri, total_ce, total_n = 0, 0, 0

        for batch_idx, (imgs, labels, _) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            embeddings, logits = model(imgs)

            l_tri = triplet_loss(embeddings, labels)
            l_ce = ce_loss(logits, labels)
            loss = l_tri + l_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_tri += l_tri.item()
            total_ce += l_ce.item()
            total_n += 1

        scheduler.step()
        avg_tri = total_tri / max(total_n, 1)
        avg_ce = total_ce / max(total_n, 1)
        lr_now = optimizer.param_groups[1]['lr']

        # Eval every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            mAP, r1, r5 = evaluate(model, query_loader, gallery_loader, device)
            print(f"[Epoch {epoch:02d}] tri={avg_tri:.4f} ce={avg_ce:.4f} lr={lr_now:.6f} | mAP={mAP:.3f} R1={r1:.3f} R5={r5:.3f}")

            if mAP > best_mAP:
                best_mAP = mAP
                # 추론용 state_dict 저장 (projection head만)
                save_path = save_dir / 'clip_veri_best.pt'
                torch.save({
                    'clip_visual': model.clip_visual.state_dict(),
                    'proj': model.proj.state_dict(),
                    'embed_dim': 512,
                    'mAP': mAP, 'R1': r1, 'R5': r5,
                    'epoch': epoch,
                }, str(save_path))
                print(f"  -> Best model 저장: {save_path} (mAP={mAP:.3f})")
        else:
            print(f"[Epoch {epoch:02d}] tri={avg_tri:.4f} ce={avg_ce:.4f} lr={lr_now:.6f}")

    print(f"\n[완료] Best mAP: {best_mAP:.3f}")
    print(f"모델 저장: {save_dir / 'clip_veri_best.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='VeRi 데이터 루트 (image_train, image_test, image_query 포함)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--p', type=int, default=16, help='P: 배치당 차량 수')
    parser.add_argument('--k', type=int, default=4, help='K: 차량당 이미지 수')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-dir', type=str, default='weights/reid')
    args = parser.parse_args()
    train(args)
