"""
학습된 모델 평가 스크립트
실행: docker exec -w /workspace HV2X python /workspace/codes/tools/training/eval_gt.py [best.pt 경로]
"""
import sys
from ultralytics import YOLO

# ── 경로 설정 ──────────────────────────────────────────────────
if len(sys.argv) > 1:
    WEIGHTS = sys.argv[1]
else:
    # 기본값: 최신 학습 결과 → 없으면 배포중 weight
    import glob, os
    ckpts = sorted(glob.glob("/workspace/outputs/runs/finetune/mixed_cyclist_hardneg*/weights/best.pt"))
    WEIGHTS = ckpts[-1] if ckpts else "/workspace/weights/yolov8_hardneg_ped.pt"

DATA_YAML = "/workspace/datasets/train_datasets/mixed_cyclist/data.yaml"
print(f"Evaluating: {WEIGHTS}")

model = YOLO(WEIGHTS)

# ── Validation (val set) ────────────────────────────────────────
print("\n--- Val set metrics ---")
metrics = model.val(
    data    = DATA_YAML,
    imgsz   = 640,
    batch   = 16,
    device  = 0,
    split   = "val",
    plots   = True,
    save_json = True,
    project = "/workspace/outputs/runs/detect",
    name    = "mixed_cyclist_hardneg_eval",
)

print(f"\nmAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")

print("\n--- Per-class AP50 ---")
names = {0: "person", 1: "cyclist", 2: "car", 3: "motorcycle", 4: "truck"}
for i, ap in enumerate(metrics.box.ap50):
    print(f"  {names.get(i, str(i)):12s}: {ap:.4f}")

print(f"\nResults saved → {metrics.save_dir}")
