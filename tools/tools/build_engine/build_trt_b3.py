import tensorrt as trt, os, sys

ONNX_PATH = 'weights/yolov8_gt_best_b3.onnx'
ENG_PATH  = 'weights/yolov8_gt_best_b3_fp16.engine'
BATCH = 3
IMGSZ = 640

print(f'TRT {trt.__version__} | ONNX: {os.path.getsize(ONNX_PATH)/1e6:.1f}MB', flush=True)

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser  = trt.OnnxParser(network, logger)

with open(ONNX_PATH, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print('Error:', parser.get_error(i))
        sys.exit(1)

print(f'Parsed. input: {network.get_input(0).name} {network.get_input(0).shape}', flush=True)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 * (1 << 30))
config.set_flag(trt.BuilderFlag.FP16)

profile = builder.create_optimization_profile()
inp = network.get_input(0).name
profile.set_shape(inp,
    min=(BATCH, 3, IMGSZ, IMGSZ),
    opt=(BATCH, 3, IMGSZ, IMGSZ),
    max=(BATCH, 3, IMGSZ, IMGSZ))
config.add_optimization_profile(profile)

print(f'Building FP16 engine batch={BATCH}... (몇 분 소요)', flush=True)
serialized = builder.build_serialized_network(network, config)
if not serialized:
    print('FAILED'); sys.exit(1)

with open(ENG_PATH, 'wb') as f:
    f.write(serialized)
print(f'Done: {ENG_PATH} ({os.path.getsize(ENG_PATH)/1e6:.1f}MB)', flush=True)
