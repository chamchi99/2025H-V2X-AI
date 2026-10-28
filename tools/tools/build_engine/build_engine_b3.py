"""
YOLOv8 모델을 batch=N, FP16 TRT engine으로 변환

사용법:
  python build_engine_b3.py --pt weights/yolov8_hardneg_ped.pt
  python build_engine_b3.py --pt weights/x.pt --onnx weights/x_b3.onnx --eng weights/x_b3_fp16.engine --batch 3 --imgsz 640

--onnx/--eng 미지정 시 --pt 이름 기준으로 자동 생성 (<stem>_b3.onnx, <stem>_b3_fp16.engine).
경로는 실행 위치(cwd) 기준 → `docker exec -w /workspace HV2X python codes/tools/build_engine/build_engine_b3.py ...` 로 실행 권장.
"""
from ultralytics import YOLO
import tensorrt as trt
import onnx
import numpy as np
import os, sys, argparse

ap = argparse.ArgumentParser(description='YOLOv8 .pt → batch=N FP16 TRT engine')
ap.add_argument('--pt', required=True, help='입력 .pt weight 경로')
ap.add_argument('--onnx', default=None, help='중간 onnx 경로 (기본: <pt stem>_b3.onnx)')
ap.add_argument('--eng', default=None, help='출력 engine 경로 (기본: <pt stem>_b3_fp16.engine)')
ap.add_argument('--batch', type=int, default=3, help='고정 배치 크기 (기본 3)')
ap.add_argument('--imgsz', type=int, default=640, help='입력 해상도 (기본 640)')
args = ap.parse_args()

PT_PATH   = args.pt
_stem     = PT_PATH[:-3] if PT_PATH.endswith('.pt') else PT_PATH
ONNX_PATH = args.onnx if args.onnx else f'{_stem}_b3.onnx'
ENG_PATH  = args.eng  if args.eng  else f'{_stem}_b3_fp16.engine'
BATCH     = args.batch
IMGSZ     = args.imgsz

if not os.path.exists(PT_PATH):
    sys.exit(f'[!] .pt 파일 없음: {PT_PATH}  (cwd={os.getcwd()})')
print(f'PT={PT_PATH}\nONNX={ONNX_PATH}\nENG={ENG_PATH}\nBATCH={BATCH} IMGSZ={IMGSZ}', flush=True)

# Step 1: Export ONNX with dynamic batch
# ⚠️ ultralytics의 CPU export가 CUDA_VISIBLE_DEVICES='' 로 덮어써서 같은 프로세스의
#    Step 2(TensorRT)가 GPU를 못 보게 됨 → export 전후로 이 env를 저장/복원한다.
print('Step 1: Export ONNX (dynamic batch)...', flush=True)
_prev_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
model = YOLO(PT_PATH)
model.export(format='onnx', dynamic=True, half=False, imgsz=IMGSZ, simplify=True)
if _prev_cvd is None:
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)   # export가 심은 '' 제거 → GPU 다시 보임
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = _prev_cvd
# ultralytics exports to same dir as .pt with .onnx extension
src_onnx = PT_PATH.replace('.pt', '.onnx')
os.rename(src_onnx, ONNX_PATH)
print(f'  ONNX saved: {ONNX_PATH} ({os.path.getsize(ONNX_PATH)/1e6:.1f}MB)', flush=True)

# Step 2: Build TRT engine with static batch=3 FP16
print(f'Step 2: Build TRT FP16 engine (batch={BATCH})...', flush=True)
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(ONNX_PATH, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f'  ONNX error: {parser.get_error(i)}')
        sys.exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 * (1 << 30))
config.set_flag(trt.BuilderFlag.FP16)

# Fix batch=3 via optimization profile
profile = builder.create_optimization_profile()
inp_name = network.get_input(0).name
profile.set_shape(inp_name,
    min=(BATCH, 3, IMGSZ, IMGSZ),
    opt=(BATCH, 3, IMGSZ, IMGSZ),
    max=(BATCH, 3, IMGSZ, IMGSZ))
config.add_optimization_profile(profile)

serialized = builder.build_serialized_network(network, config)
if serialized is None:
    print('Build failed!')
    sys.exit(1)

with open(ENG_PATH, 'wb') as f:
    f.write(serialized)
print(f'  Engine saved: {ENG_PATH} ({os.path.getsize(ENG_PATH)/1e6:.1f}MB)', flush=True)
print('Done!', flush=True)
