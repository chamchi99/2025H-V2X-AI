"""
현재 사용중 weight(yolov8_hardneg_ped.pt) 재현 학습 스크립트
  base: yolov8_gt_best.pt → mixed_cyclist 데이터셋(hardneg 보행자 포함) fine-tuning
실행: docker exec HV2X python /workspace/codes/tools/training/train_gt.py
"""
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────
#WEIGHTS   = "/workspace/weights/yolov8_gt_best.pt"                              # base
#DATA_YAML = "/workspace/datasets/train_datasets/mixed_cyclist/data.yaml"        # cyclist 통합 + hardneg ped
#PROJECT   = "/workspace/outputs/runs/finetune"
#NAME      = "mixed_cyclist_hardneg"

WEIGHTS   = "/workspace/weights/yolov8_hardneg_ped.pt"                              # base
DATA_YAML = "/workspace/datasets/train_datasets/mixed_cyclist_reproduced_laneoccadded/dataset.yaml"        # cyclist 통합 + hardneg ped
PROJECT   = "/workspace/outputs/runs/finetune"
NAME      = "mixed_cyclist_hardneg_260716A_handover"

model = YOLO(WEIGHTS)

results = model.train(
    data        = DATA_YAML,
    epochs      = 40,
    imgsz       = 640,
    batch       = 16,
    device      = 0,           # GPU 0 (Quadro RTX 8000)
    workers     = 8,
    project     = PROJECT,
    name        = NAME,
    exist_ok    = False,
    # Fine-tuning 하이퍼파라미터 (yolov8_hardneg_ped.pt 재현)
    freeze      = 10,          # backbone freeze (top-down 유지)
    lr0         = 0.0005,      # 초기 LR (pretrained이므로 낮게)
    lrf         = 0.01,        # 최종 LR = lr0 * lrf
    warmup_epochs = 3,
    patience    = 15,          # early stopping
    val         = True,
    plots       = True,
    # Augmentation
    hsv_h       = 0.015,
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    degrees     = 0.0,
    translate   = 0.1,
    scale       = 0.5,
    flipud      = 0.0,
    fliplr      = 0.5,
    mosaic      = 1.0,
    mixup       = 0.1,
)

print("\n=== Training complete ===")
print(f"Best model: {results.save_dir}/weights/mixed_cyclist_laneocc_best_260716A.pt")
