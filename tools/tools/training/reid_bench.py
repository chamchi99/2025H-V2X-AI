import time, numpy as np, torch, sys
from pathlib import Path
sys.path.append('/workspace')
sys.path.append('/workspace/trackers/strong_sort')
from trackers.strong_sort.reid_multibackend import ReIDDetectMultiBackend

device = torch.device('cuda:0')
clf = ReIDDetectMultiBackend(Path('/workspace/weights/osnet_x1_0_vehicle.pt'), device=device)
crops5 = [np.random.randint(0,255,(80,60,3),dtype=np.uint8) for _ in range(5)]
crops3 = [np.random.randint(0,255,(80,60,3),dtype=np.uint8) for _ in range(3)]
for _ in range(10): clf(crops5)
t5, t3 = [], []
for _ in range(50):
    s = time.time(); clf(crops5); t5.append((time.time()-s)*1000)
    s = time.time(); clf(crops3); t3.append((time.time()-s)*1000)
avg5 = sum(t5)/len(t5)
avg3 = sum(t3)/len(t3)
print(f'5crops(cam2): {avg5:.1f}ms')
print(f'3crops(cam3/cam1): {avg3:.1f}ms')
print(f'3카메라 합산(AV未발견시): {avg5 + 2*avg3:.1f}ms/frame')
