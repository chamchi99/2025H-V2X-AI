import tensorrt as trt
import os, sys

onnx_path = 'weights/yolov8_gt_best.onnx'
engine_path = 'weights/yolov8_gt_best_trt10.engine'

print(f'TRT version: {trt.__version__}', flush=True)

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f'ONNX error: {parser.get_error(i)}')
        sys.exit(1)

print(f'ONNX parsed OK. inputs:{network.num_inputs} outputs:{network.num_outputs}', flush=True)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
config.set_flag(trt.BuilderFlag.FP16)

print('Building TRT FP16 engine... (may take 3-10 mins)', flush=True)
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    print('Build failed!')
    sys.exit(1)

with open(engine_path, 'wb') as f:
    f.write(serialized)

print(f'Done! {engine_path} ({os.path.getsize(engine_path)/1e6:.1f}MB)', flush=True)
