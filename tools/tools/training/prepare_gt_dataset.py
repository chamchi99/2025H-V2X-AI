"""
gt_dataset 준비 스크립트
- train/val 80/20 분리
- dataset.yaml 생성
"""
import os
import shutil
import random

#SRC_IMAGES = "/home/super/gt_dataset/images"
#SRC_LABELS = "/home/super/gt_dataset/labels"
#DST_BASE   = "/workspace/gt_dataset"

SRC_IMAGES = "/workspace/datasets/train_datasets/mixed_cyclist_reproduced_laneoccadded/images/total"
SRC_LABELS = "/workspace/datasets/train_datasets/mixed_cyclist_reproduced_laneoccadded/labels/total"
DST_BASE   = "/workspace/datasets/train_datasets/mixed_cyclist_reproduced_laneoccadded"

# Class remap: 7(truck) → 4, others unchanged
REMAP = {7: 4}
NC = 5
NAMES = {0: "person", 1: "cyclist", 2: "car", 3: "motorcycle", 4: "truck"}

random.seed(42)

# ── 파일 목록 수집 ──────────────────────────────────────────────
img_files = sorted([f for f in os.listdir(SRC_IMAGES) if f.endswith(('.jpg', '.png'))])
print(f"Total images: {len(img_files)}")

# 라벨 없는 이미지 제외
valid = []
for img in img_files:
    stem = os.path.splitext(img)[0]
    lbl = os.path.join(SRC_LABELS, stem + ".txt")
    if os.path.exists(lbl):
        valid.append(stem)
    else:
        print(f"  [SKIP] no label: {img}")

print(f"Valid pairs: {len(valid)}")

# ── Train/Val 분리 ──────────────────────────────────────────────
random.shuffle(valid)
n_val = max(1, int(len(valid) * 0.1)) # 여기가 validation 분리하는 비율임 
val_set   = set(valid[:n_val])
train_set = set(valid[n_val:])
print(f"Train: {len(train_set)}, Val: {len(val_set)}")

# ── 디렉터리 생성 ───────────────────────────────────────────────
for split in ("train", "val"):
    os.makedirs(f"{DST_BASE}/images/{split}", exist_ok=True)
    os.makedirs(f"{DST_BASE}/labels/{split}", exist_ok=True)

# ── 파일 복사 + 라벨 리매핑 ────────────────────────────────────
def copy_and_remap(stems, split):
    for stem in stems:
        # 이미지
        for ext in (".jpg", ".png", ".jpeg"):
            src_img = os.path.join(SRC_IMAGES, stem + ext)
            if os.path.exists(src_img):
                shutil.copy2(src_img, f"{DST_BASE}/images/{split}/{stem}{ext}")
                break

        # 라벨 (클래스 리매핑)
        src_lbl = os.path.join(SRC_LABELS, stem + ".txt")
        dst_lbl = f"{DST_BASE}/labels/{split}/{stem}.txt"
        with open(src_lbl, "r") as fin, open(dst_lbl, "w") as fout:
            for line in fin:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                cls = REMAP.get(cls, cls)
                fout.write(f"{cls} {' '.join(parts[1:])}\n")

copy_and_remap(train_set, "train")
copy_and_remap(val_set,   "val")
print("Files copied and remapped.")

# ── dataset.yaml 생성 ──────────────────────────────────────────
yaml_content = f"""path: {DST_BASE}
train: images/train
val: images/val

nc: {NC}
names:
"""
for idx, name in NAMES.items():
    yaml_content += f"  {idx}: {name}\n"

yaml_path = f"{DST_BASE}/dataset.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)
print(f"dataset.yaml written → {yaml_path}")
print("\nDone! Ready for training.")
