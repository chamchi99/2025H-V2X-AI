import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import math
import time
import json
import logging
import cv2

##########정윤성
import socket
import threading
import queue
##########

from collections import deque
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from PIL import Image

from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT.parent / 'weights'

# 실행 위치(cwd)에 의존하지 않도록 코드 폴더로 cwd 고정.
# main.py·트래커가 'trackers/.../*.yaml', 출력 json 등을 cwd 상대경로로 참조하므로,
# 어디서 실행하든(-w 없이도) 동작하게 함.
# 단 --source 등 사용자가 준 상대경로는 '원래 실행 위치' 기준이어야 하므로 여기서 기억.
_ORIG_CWD = os.getcwd()
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (check_img_size, check_requirements,
                                  check_imshow, increment_path,
                                  strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo_comb_v3 import create_tracker
from trackers.strong_sort.reid_multibackend import ReIDDetectMultiBackend

import geopy.distance as geopydist
from speed import *
from ch_setting_comb import *

# CLIP+VeRi 기반 ReIDExtractor를 re-id.py에서 동적 로드 (파일명 hyphen 때문)
import importlib.util as _ilu
_reid_spec = _ilu.spec_from_file_location("_reid_module", str(ROOT / "re-id.py"))
_reid_module = _ilu.module_from_spec(_reid_spec)
_reid_spec.loader.exec_module(_reid_module)
ReIDExtractor = _reid_module.ReIDExtractor


class CAVReID:
    """CAV(목표차량) cross-camera Re-ID.
    - CLIP+VeRi feature 추출
    - multi-query max similarity 매칭
    - threshold 기반 이진 판별 (CAV냐 아니냐)
    """
    def __init__(self, query_dir, threshold=0.75, device='cuda', extractor=None):
        self.threshold = threshold
        # extractor 공유(여러 타깃이 같은 CLIP+VeRi 추출기 재사용) → CLIP 중복 로드 방지
        self.extractor = extractor if extractor is not None else ReIDExtractor(weights=None, device=device)
        self.query_features = []  # list of (D,) np.ndarray, 개별 보존 (max similarity용)
        self._load_queries(query_dir)

    def _load_queries(self, query_dir):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        qdir = Path(query_dir)
        if not qdir.exists():
            raise FileNotFoundError(f"CAV query 폴더 없음: {qdir}")
        files = sorted([f for f in qdir.iterdir() if f.suffix.lower() in exts])
        if not files:
            raise FileNotFoundError(f"CAV query 폴더에 이미지 없음: {qdir}")
        crops = []
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                crops.append(img)
        feats = self.extractor.extract_from_crops(crops)
        self.query_features = [feats[i] for i in range(len(feats))]
        print(f"[CAVReID] CLIP+VeRi 쿼리 {len(self.query_features)}장 로드: {[f.name for f in files]} (threshold={self.threshold})")

    def best_sim(self, crops):
        """crop 리스트에 대해 각각의 max-similarity 반환.
        Returns: np.ndarray (N,) — 각 crop에 대한 모든 쿼리 중 최대 sim
        """
        if len(crops) == 0 or len(self.query_features) == 0:
            return np.array([])
        feats = self.extractor.extract_from_crops(crops)  # (N, D), L2-normed
        return self.best_sim_feats(feats)

    def best_sim_feats(self, feats):
        """미리 추출한 crop feature로 max-similarity 계산 (여러 타깃이 추출 1회 공유)."""
        if len(feats) == 0 or len(self.query_features) == 0:
            return np.array([])
        # max over queries: shape (N,)
        sims = np.stack([feats @ qf for qf in self.query_features], axis=0)  # (Q, N)
        return sims.max(axis=0)

    def is_cav_match(self, sims, prev_best=0.0):
        """sims 중 threshold 통과하고 prev_best보다 높은 best 반환.
        Returns: (best_idx, best_sim) or (None, prev_best)
        """
        if len(sims) == 0:
            return None, prev_best
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= self.threshold and best_sim > prev_best:
            return best_idx, best_sim
        return None, prev_best

LOGGER = logging.getLogger(__name__)
if logging.getLogger().handlers:
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])

REF_LAT, REF_LON = 36.38234, 127.3667
SPEED_MAX_KMH = 200.0
SPEED_MIN_KMH = 3.0


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def yolo_to_det(results, device='cuda'):
    """Convert YOLO results to detection tensor [x1,y1,x2,y2,conf,cls]"""
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return torch.zeros((0, 6), device=device)
    return torch.cat([boxes.xyxy, boxes.conf.unsqueeze(1), boxes.cls.unsqueeze(1)], dim=1)


class TRTInfer:
    """TRT engine batch=3 direct inference. NMS-free output (B,300,6)."""
    def __init__(self, engine_path, conf=0.5, iou=0.45, device='cuda:0'):
        import tensorrt as trt
        self.conf = conf
        self.iou = iou
        self.dev = torch.device(device)
        self.imgsz = 640
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.inp_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        print(f'[TRT] engine loaded: {engine_path}', flush=True)

    def preprocess(self, imgs):
        self.orig_shapes = []
        batch = []
        for img in imgs:
            self.orig_shapes.append(img.shape[:2])
            r = cv2.resize(img, (self.imgsz, self.imgsz))
            r = r[:, :, ::-1].copy().astype(np.float32) / 255.0
            batch.append(r)
        arr = np.stack(batch, 0).transpose(0, 3, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.dev)

    def postprocess(self, raw, orig_shapes, conf_thres):
        results = []
        for b in range(raw.shape[0]):
            pred = raw[b].float()
            pred = pred[pred[:, 4] >= conf_thres]
            if len(pred) == 0:
                results.append(torch.zeros((0, 6), device=self.dev))
                continue
            oh, ow = orig_shapes[b]
            pred[:, 0] *= ow / self.imgsz; pred[:, 2] *= ow / self.imgsz
            pred[:, 1] *= oh / self.imgsz; pred[:, 3] *= oh / self.imgsz
            results.append(torch.cat([pred[:, :4],
                                      pred[:, 4].unsqueeze(1),
                                      pred[:, 5].unsqueeze(1)], dim=1))
        return results

    @torch.no_grad()
    def __call__(self, imgs, conf_thres=None, iou_thres=None):
        conf_thres = conf_thres or self.conf
        inp = self.preprocess(imgs)
        B = inp.shape[0]
        out_tensor = torch.zeros((B, 300, 6), dtype=torch.float32, device=self.dev)
        self.ctx.set_tensor_address(self.inp_name, inp.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out_tensor.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return self.postprocess(out_tensor, self.orig_shapes, conf_thres)


def calc_heading_speed(lat_now, lon_now, lat_prev, lon_prev, frame_gap, fps=30.0):
    dlat = lat_now - lat_prev
    dlon = lon_now - lon_prev
    dist_m = math.sqrt((dlat * 111320)**2 + (dlon * 111320 * math.cos(math.radians(lat_now)))**2)
    dt_sec = frame_gap / fps
    speed_kmh = (dist_m / dt_sec) * 3.6 if dt_sec > 0 else 0
    heading_rad = math.atan2(dlat, dlon * math.cos(math.radians(lat_now)))
    return heading_rad, speed_kmh


def smooth_heading_speed(track_id, heading_raw, speed_raw, smooth_data, smooth_len=6):
    if speed_raw > SPEED_MAX_KMH:
        if track_id in smooth_data and len(smooth_data[track_id]["speed"]) > 0:
            sd = smooth_data[track_id]
            avg_speed = round(sum(sd["speed"]) / len(sd["speed"]), 1)
            avg_heading = math.atan2(sum(sd["hy"]) / len(sd["hy"]),
                                     sum(sd["hx"]) / len(sd["hx"]))
            return avg_heading, avg_speed
        return heading_raw, 0.0
    if track_id not in smooth_data:
        smooth_data[track_id] = {
            "speed": deque(maxlen=smooth_len),
            "hx": deque(maxlen=smooth_len),
            "hy": deque(maxlen=smooth_len)
        }
    sd = smooth_data[track_id]
    sd["speed"].append(speed_raw)
    sd["hx"].append(math.cos(heading_raw))
    sd["hy"].append(math.sin(heading_raw))
    avg_speed = round(sum(sd["speed"]) / len(sd["speed"]), 1)
    avg_heading = math.atan2(sum(sd["hy"]) / len(sd["hy"]),
                             sum(sd["hx"]) / len(sd["hx"]))
    if avg_speed < SPEED_MIN_KMH:
        avg_speed = 0.0
    return avg_heading, avg_speed


def _update_stop(id, cam_idx, speed_val, stop_start, stop_moving_frames,
                 STOP_RESET_FRAMES, SPEED_MIN_KMH, frame_idx=None, video_fps=30.0):
    """정차 시간 계산.
    frame_idx가 주어지면 video time (frame_idx / video_fps) 기준,
    아니면 wall clock 기준 (구버전 호환).
    """
    _skey = (cam_idx, id)
    # 시간 단위: frame_idx 있으면 video time, 없으면 wall clock
    if frame_idx is not None:
        _now = frame_idx / video_fps
    else:
        _now = time.time()
    if speed_val != 999 and speed_val < SPEED_MIN_KMH:
        stop_moving_frames.pop(_skey, None)
        if _skey not in stop_start:
            stop_start[_skey] = _now
        return int(_now - stop_start[_skey])
    else:
        stop_moving_frames[_skey] = stop_moving_frames.get(_skey, 0) + 1
        if stop_moving_frames[_skey] >= STOP_RESET_FRAMES:
            stop_start.pop(_skey, None)
            stop_moving_frames.pop(_skey, None)
            return 0
        return int(_now - stop_start[_skey]) if _skey in stop_start else 0


def toWGS(x1, y1):
    # [임시2] gcp/temp 14 GCP 재캘리브 — RMS 0.214m (2026-07-21). 원본: main.py.pre_temph2_bak
    H = [[ 4.81884372e-02,  1.46872518e-01, -3.90567918e+00],
         [-2.12287447e-02, -1.41711086e-01,  1.33550456e+02],
         [-8.44830080e-04,  1.26793175e-02,  1.00000000e+00]]
    R1 = np.matmul(H, [x1, y1, 1])
    R1 /= R1[2]
    return np.array([REF_LAT + R1[1] / 111320,
                     REF_LON + R1[0] / (111320 * np.cos(np.radians(REF_LAT))),
                     1.0])


def toWGS_ch1(x1, y1):
    # [임시2] gcp/temp 9 GCP 재캘리브 — RMS 0.570m (2026-07-21). 원본: main.py.pre_temph2_bak
    H = [[ 3.19076673e-03,  3.87161216e-02, -3.19400988e+01],
         [ 1.67473317e-02, -1.00352190e-02, -1.27416365e+01],
         [-1.88681221e-04,  2.47942650e-03,  1.00000000e+00]]
    R1 = np.matmul(H, [x1, y1, 1])
    R1 /= R1[2]
    return np.array([REF_LAT + R1[1] / 111320,
                     REF_LON + R1[0] / (111320 * np.cos(np.radians(REF_LAT))),
                     1.0])


def bbox_to_gps_polygon(x1, y1, x2, y2, towgs_fn):
    """bbox 4코너를 GPS로 투영해 polygon (closed ring) 반환.
    Returns: [[lat,lon], [lat,lon], [lat,lon], [lat,lon], [lat,lon]] (TL→TR→BR→BL→TL)
    """
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    ring = []
    for px, py in corners:
        g = towgs_fn(px, py)
        ring.append([float(g[0]), float(g[1])])
    ring.append(ring[0])  # close
    return ring


# 클래스별 표준 footprint 크기 (length forward, width side) — 미터
CLASS_FOOTPRINT_M = {
    0: (0.6, 0.6),   # person
    1: (1.8, 0.6),   # cyclist
    2: (4.5, 1.8),   # car
    3: (2.0, 0.8),   # motorcycle
    4: (7.0, 2.5),   # truck
}


def compute_road_heading(towgs_fn, x_center=960.0, y_far=200.0, y_near=900.0):
    """카메라의 도로 방향을 호모그래피로 계산.
    이미지 세로 가운데 선(far→near)을 GPS로 투영해 방향 벡터를 atan2로 변환.
    정차차량처럼 movement heading이 noise인 경우 default로 사용.
    Returns: heading_rad (atan2 표준)
    """
    g_far = towgs_fn(x_center, y_far)
    g_near = towgs_fn(x_center, y_near)
    dlat = g_near[0] - g_far[0]
    dlon = g_near[1] - g_far[1]
    cos_lat = math.cos(math.radians(g_near[0]))
    return math.atan2(dlat, dlon * cos_lat)


def gps_oriented_rect(lat, lon, heading_rad, cls):
    """차량 footprint를 GPS 좌표계에서 heading 따라 회전한 직사각형 5점 반환.
    heading_rad: math.atan2(dlat, dlon*cos(lat)) 기준 (atan2 표준)
    Returns: [[lat,lon] × 5] (closed ring)
    """
    L, W = CLASS_FOOTPRINT_M.get(int(cls), (4.5, 1.8))
    # heading 무효 시 기본 0 (동쪽)
    if heading_rad is None or heading_rad == 999:
        heading_rad = 0.0
    cos_h = math.cos(heading_rad); sin_h = math.sin(heading_rad)
    # local: (forward, side), CCW order
    corners_local = [(L/2, -W/2), (L/2, W/2), (-L/2, W/2), (-L/2, -W/2)]
    ring = []
    cos_lat = math.cos(math.radians(lat))
    for fx, sy in corners_local:
        dx_m = fx * cos_h - sy * sin_h  # east offset (m)
        dy_m = fx * sin_h + sy * cos_h  # north offset (m)
        d_lat = dy_m / 111320.0
        d_lon = dx_m / (111320.0 * cos_lat) if cos_lat else 0.0
        ring.append([lat + d_lat, lon + d_lon])
    ring.append(ring[0])
    return ring


_SDSM_CN = ["person", "cyclist", "car", "moto", "truck"]

def print_sdsm(traffic, frame_idx, ai_latency):
    """실제 산출되는 SDSM-G(cctv-traffic) 내용을 C++ 수신기 대시보드와 동일 포맷으로 출력."""
    print(f"Frame {frame_idx:<7d}  objects: {len(traffic):<3d}  AI latency: {ai_latency*1000:6.1f} ms\n")
    print("cam class    id     lat, lon              spd  head  lane  zone       stop/occ")
    print("─" * 83)
    for o in traffic:
        c = int(o["class"]); cls = _SDSM_CN[c] if 0 <= c < 5 else "?"
        _id = o["id"]
        idstr = "AV" if _id == 88888 else ("CV" if _id == 77777 else str(_id))
        head_deg = o["heading"] * 180.0 / math.pi
        zone = o["ped_zone"] if o.get("ped_zone") else "-"
        so = f"{o['stop_duration']}s" if o.get("stop_duration", 0) > 0 else ""
        occ = o.get("lane_occupancy")
        if isinstance(occ, list):
            for i, v in enumerate(occ):
                if v > 0:
                    so += f" L{i+1}:{int(v*100)}%"
        if not so:
            so = "-"
        lat, lon = o["GPS"][0], o["GPS"][1]
        print(f" {int(o['cctv']):2d} {cls:<7s} {idstr:>5s}  {lat:9.6f},{lon:10.6f} "
              f"{o['speed']:5.1f} {head_deg:5.0f} {int(o['lane']):4d}  {zone:<9s} {so}")
        bg = o.get("bbox_gps")
        if isinstance(bg, list) and bg:
            line = "    └ bbox_gps:"
            for p in bg[:4]:
                line += f" ({p[0]:.5f},{p[1]:.5f})"
            print(line)
    print("─" * 83)


def toWGS_ch3(x1, y1):
    # [임시2] gcp/temp 10 GCP 재캘리브 — RMS 0.098m (2026-07-21). 원본: main.py.pre_temph2_bak
    H = [[ 6.20174532e-03,  8.98854220e-03,  7.75180224e+00],
         [-1.30266816e-03, -1.28218238e-02,  1.64351185e+00],
         [ 6.89762466e-06,  8.24020159e-04,  1.00000000e+00]]
    R1 = np.matmul(H, [x1, y1, 1])
    R1 /= R1[2]
    return np.array([REF_LAT + R1[1] / 111320,
                     REF_LON + R1[0] / (111320 * np.cos(np.radians(REF_LAT))),
                     1.0])


def _convex_hull(points):
    """Andrew's monotone chain. points: list[(x,y)] → hull verts CCW."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def compute_camera_roi(towgs_fn, img_w=1920, img_h=1080, step=20,
                       ref_lat=None, max_dist_m=250.0):
    """카메라가 GPS상에서 커버하는 영역을 homography로 계산해 convex hull로 반환.
    이미지 그리드를 GPS로 투영하고, REF(ref_lat)에서 max_dist_m 넘는 발산점은 제외.
    Returns: list[(lat, lon)] convex hull (점이 부족하면 []).
    """
    ref_lat = ref_lat if ref_lat is not None else REF_LAT
    cos_lat = math.cos(math.radians(ref_lat))
    pts_m, latlon = [], []
    for px in range(0, img_w + 1, step):
        for py in range(0, img_h + 1, step):
            g = towgs_fn(px, py)
            lat, lon = float(g[0]), float(g[1])
            e = (lon - REF_LON) * 111320.0 * cos_lat
            n = (lat - REF_LAT) * 111320.0
            if math.hypot(e, n) <= max_dist_m:
                pts_m.append((e, n)); latlon.append((lat, lon))
    if len(pts_m) < 3:
        return []
    hull_m = _convex_hull(pts_m)
    # meter hull 정점을 다시 lat/lon으로 (역변환 대신 매핑)
    m2ll = {(round(e, 4), round(nn, 4)): (lat, lon)
            for (e, nn), (lat, lon) in zip(pts_m, latlon)}
    hull_ll = []
    for e, nn in hull_m:
        key = (round(e, 4), round(nn, 4))
        if key in m2ll:
            hull_ll.append(m2ll[key])
        else:
            hull_ll.append((REF_LAT + nn / 111320.0,
                            REF_LON + e / (111320.0 * cos_lat)))
    return hull_ll


def point_in_poly(lat, lon, poly):
    """Ray casting. poly: list[(lat, lon)] (closed 불필요)."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = poly[i]      # (lat, lon)
        yj, xj = poly[j]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


# overlap dedup ROI (맵에서 수동 지정, 2026-06-24): 이 폴리곤 안에선 cam3(zoom)을
# 버리고 cam2를 채택(cam2가 가장자리에서 더 정확). cam3은 이 영역 밖(유니크 영역)에서만 표출.
OVERLAP_ROI = [
    (36.3822915, 127.3669218),
    (36.3822938, 127.3669155),
    (36.3822980, 127.3669050),
    (36.3823259, 127.3668358),
    (36.3823270, 127.3668340),
    (36.3823362, 127.3668312),
    (36.3823738, 127.3668331),
    (36.3823753, 127.3668334),
    (36.3824920, 127.3669032),
    (36.3825667, 127.3669560),
    (36.3825648, 127.3669583),
    (36.3825617, 127.3669590),
    (36.3825597, 127.3669594),
    (36.3823446, 127.3669935),
]


def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)


def get_ped_zone(foot_x, foot_y, crosswalk_poly, sidewalk_poly, road_poly=None):
    pt = (int(foot_x), int(foot_y))
    if crosswalk_poly is not None and cv2.pointPolygonTest(crosswalk_poly, pt, False) >= 0:
        return "crosswalk"
    if sidewalk_poly is not None and cv2.pointPolygonTest(sidewalk_poly, pt, False) >= 0:
        return "sidewalk"
    if road_poly is None:
        return "road"
    if cv2.pointPolygonTest(road_poly, pt, False) >= 0:
        return "road"
    return "unknown"


##########정윤성
#------------------TCP Stream Server-----------
class YoloTCPServer:
    def __init__(self, host='0.0.0.0', port=9000, max_queue_size=30):
        self.host = host
        self.port = port
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.client_connected = threading.Event()
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

    def push_result(self, json_data: dict):
        try:
            if self.data_queue.full():
                self.data_queue.get_nowait()
            self.data_queue.put_nowait(json_data)
        except queue.Full:
            pass

    def _server_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        while self.running:
            try:
                conn, addr = srv.accept()
                self._handle_client(conn)
            except Exception as e:
                pass

    def _handle_client(self, conn):
        self.client_connected.set()
        try:
            while self.running:
                try:
                    data_dict = self.data_queue.get(timeout=1.0)
                except queue.Empty:
                    continue   # 큐가 잠깐 비어도 연결 유지 (끊지 않음)
                json_str = json.dumps(data_dict, ensure_ascii=False, default=str) + "\n"
                conn.sendall(json_str.encode('utf-8'))
        except (socket.error, BrokenPipeError):
            pass   # 진짜 연결 끊김만 처리
        finally:
            conn.close()
            self.client_connected.clear()
            with self.data_queue.mutex:
                self.data_queue.queue.clear()
##########--------------------------------------------------------############


# ─────────────────────────────────────────────────────────────────────
# BEV 주행경로 기반 cross-camera ReID + 겹침 dedup (실험: --bev-reid)
#   - 모든 검출을 GPS(BEV) 공간에서 단일 global track으로 통합
#   - 매 프레임: ① 동일객체 클러스터링(겹침 dedup) ② 궤적 예측 후 association
#   - 운동모델은 _predict()만 교체하면 CV → CTRV/맵 기반으로 업그레이드 가능
# ─────────────────────────────────────────────────────────────────────
class CrossCameraLinker:
    """주행경로 기반 cross-camera ReID (시간 누적 + 락).

    단일카메라 트래커(StrongSORT/OC-SORT)가 준 (cam,id)를 '안정 손잡이'로 삼아,
    서로 다른 카메라의 트랙쌍이 여러 프레임 GPS근접+동시진행하면 점수를 쌓고,
    임계를 넘으면 '동일 객체'로 락 → 겹침 dedup을 일관 적용(per-frame 흔들림 없음).
    거리는 가까울수록 큰 가중치(같은차 빠르게, 옆차선은 느리게/감쇠). 외형 미사용(v1).
    """
    def __init__(self, fps=30.0, gate_m=4.0, lock_th=5.0, score_cap=12.0,
                 max_age=20, cam_priority=None, overlap_dedup_m=5.0, anchor_hold=12):
        self.fps = fps
        self.gate_m = gate_m          # cross-camera 쌍 후보로 볼 GPS 거리
        self.overlap_dedup_m = overlap_dedup_m  # cam2↔cam3 겹침: 이 거리내면 즉시 동일차 통합
        self.lock_th = lock_th        # 누적점수 이 이상이면 동일객체 확정(락)
        self.score_cap = score_cap
        self.max_age = max_age        # (cam,id) 미관측 N프레임 후 정리
        self.cam_prio = cam_priority or {3: 0, 2: 1, 1: 2}  # cam3(zoom) 우선
        self.score = {}               # frozenset({keyA,keyB}) -> float
        self.parent = {}              # union-find: key -> parent (락된 동일객체 그룹)
        self.gid = {}                 # root key -> global_id
        self.next_gid = 1
        self.last_seen = {}           # key -> frame_idx
        self.gcams = {}               # gid -> 그 그룹이 본 카메라 set (멀티캠 판정)
        self.anchor = {}              # gid -> {'cam', 'last'} 대표 카메라 고정(ABABAB 방지)
        self.anchor_hold = anchor_hold  # 앵커 잠깐 없을 때 다른캠 억제 허용 프레임(맵 ghost로 coast)

    @staticmethod
    def _grp(c):
        return 0 if c == 0 else (1 if c in (1, 3) else 2)

    @staticmethod
    def _dm(la1, lo1, la2, lo2):
        dn = (la1 - la2) * 111320.0
        de = (lo1 - lo2) * 111320.0 * math.cos(math.radians(la1))
        return math.hypot(dn, de)

    def _find(self, k):
        p = self.parent.get(k, k)
        while p != k:
            k = p; p = self.parent.get(k, k)
        return k

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        ga, gb = self.gid.get(ra), self.gid.get(rb)
        # 더 오래된(작은) gid를 대표로 유지
        if gb is not None and (ga is None or gb < ga):
            self.parent[ra] = rb
            self.gid.pop(ra, None)
        else:
            self.parent[rb] = ra
            self.gid.pop(rb, None)

    def _gid_of(self, k):
        r = self._find(k)
        if r not in self.gid:
            self.gid[r] = self.next_gid; self.next_gid += 1
        return self.gid[r]

    def update(self, dets, frame_idx):
        """dets: [{key:(cam,id), lat,lon, cls, cam, bbox_area}]
        반환: (assign {det_idx -> global_id}, drop {det_idx})"""
        assign, drop = {}, set()
        n = len(dets)
        for d in dets:
            self.last_seen[d['key']] = frame_idx

        # 1) cross-camera 쌍 점수 누적(가까울수록 ↑) / 멀어지면 감쇠 → 임계 넘으면 락
        seen_pair = set()
        for i in range(n):
            for j in range(i + 1, n):
                a, b = dets[i], dets[j]
                if a['cam'] == b['cam'] or a['key'] == b['key']:
                    continue
                if self._grp(a['cls']) != self._grp(b['cls']):
                    continue
                pk = frozenset((a['key'], b['key']))
                seen_pair.add(pk)
                dist = self._dm(a['lat'], a['lon'], b['lat'], b['lon'])
                if {a['cam'], b['cam']} == {2, 3} and dist <= self.overlap_dedup_m:
                    # cam2↔cam3 겹침구간: 같은 위치 = 같은 차 → 즉시 통합(시간누적 락 불필요).
                    # per-camera id-switch에 무관. 한쪽이 CAV면 통합 대표가 AV 라벨 보존(cav_gids).
                    self.score[pk] = self.score_cap
                    self._union(a['key'], b['key'])
                elif dist <= self.gate_m:
                    w = (self.gate_m - dist) / self.gate_m  # 0~1, 가까울수록 큼
                    self.score[pk] = min(self.score_cap, self.score.get(pk, 0.0) + w)
                    if self.score[pk] >= self.lock_th:
                        self._union(a['key'], b['key'])
                else:
                    self.score[pk] = self.score.get(pk, 0.0) - 0.5  # 게이트 밖 → 감쇠

        # 둘 다 봤는데 후보쌍에 안 든(=멀어진) 경우도 감쇠 처리됨(위 else). 0 이하 제거
        for pk in [p for p, v in self.score.items() if v <= 0]:
            del self.score[pk]

        # 2) global_id 부여 (락된 그룹은 공유, 아니면 트랙별 고유)
        for di, d in enumerate(dets):
            assign[di] = self._gid_of(d['key'])

        # 3) 같은 global_id 동시 출현(락된 동일객체) → 대표 1개만, 나머지 drop.
        #    겹침 그룹(멀티캠)은 '앵커 카메라'에 고정 → cam2↔cam3 대표 교대(ABABAB) 방지.
        members = {}
        for di in range(n):
            members.setdefault(assign[di], []).append(di)
        for g, ms in members.items():
            cams_now = {dets[m]['cam'] for m in ms}
            gc = self.gcams.setdefault(g, set()); gc |= cams_now   # 그룹이 본 카메라 누적
            anc = self.anchor.get(g)
            if anc is not None and anc['cam'] in cams_now:
                # 앵커 카메라가 이번 프레임 있음 → 그걸 대표로 (위치 고정)
                rep = max((m for m in ms if dets[m]['cam'] == anc['cam']),
                          key=lambda m: dets[m]['bbox_area'])
                anc['last'] = frame_idx
            elif (len(gc) >= 2 and anc is not None
                  and frame_idx - anc['last'] <= self.anchor_hold):
                # 멀티캠 그룹인데 앵커가 잠깐 없음(최근 봤음) → 전부 억제 → 맵 ghost가 앵커 위치 유지
                for m in ms:
                    drop.add(m)
                continue
            else:
                # 앵커 없음/만료 → cam_prio로 (재)선정 후 앵커 갱신
                rep = min(ms, key=lambda m: (self.cam_prio.get(dets[m]['cam'], 9), -dets[m]['bbox_area']))
                self.anchor[g] = {'cam': dets[rep]['cam'], 'last': frame_idx}
            for m in ms:
                if m != rep:
                    drop.add(m)

        self._cleanup(frame_idx)
        return assign, drop

    def _cleanup(self, frame_idx):
        dead = [k for k, f in self.last_seen.items() if frame_idx - f > self.max_age]
        for k in dead:
            del self.last_seen[k]
            self.parent.pop(k, None)
            self.gid.pop(k, None)
        for pk in [p for p in self.score
                   if any(k not in self.last_seen for k in p)]:
            del self.score[pk]
        # 사라진 그룹의 앵커/카메라 기록 정리
        active_gids = set(self.gid.values())
        for g in [g for g in self.gcams if g not in active_gids]:
            self.gcams.pop(g, None); self.anchor.pop(g, None)


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov8_hardneg_ped.pt',
        reid_weights=WEIGHTS / 'osnet_x1_0_vehicle.pt',
        tracking_method='strongsort',
        imgsz=(640, 640),
        conf_thres=0.5,
        iou_thres=0.45,
        max_det=1000,
        device='',
        show_vid=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        save_vid=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=WEIGHTS.parent / 'outputs' / 'runs' / 'track',
        name='exp',
        exist_ok=False,
        line_thickness=2,
        hide_labels=False,
        hide_conf=False,
        hide_class=False,
        half=False,
        dnn=False,
        cav_query=str(WEIGHTS.parent / 'datasets' / 'reid_queries' / 'query_image'),
        cav_threshold=0.75,
        cav_id=88888,
        cv_reid=False,   # True면 CV(연결차) Re-ID도 함께 실행 (기본 off=CAV만)
        cv_query=str(WEIGHTS.parent / 'datasets' / 'reid_queries' / 'cv_query_image'),
        cv_threshold=0.78,
        cv_id=77777,
        reid_clip_weight=None,   # CLIP:색상 비중 override (None=기본 0.8)
        show_sdsm=True,          # 산출 SDSM-G(cctv-traffic) 내용 콘솔 출력 (기본 ON, --no-print-sdsm으로 끔)
        frame_skip=1,  # 1=모든 프레임 처리, N=N프레임마다 1번 처리 (real-time pacing용)
        realtime=False,  # True면 영상 파일을 RTSP처럼 처리 (wall time보다 뒤처진 frame은 자동 skip)
        bev_reid=True,   # 기본 ON: BEV 주행경로 기반 cross-camera ReID+겹침 dedup (재캘리브 GPS 기준). --no-bev-reid로 끔
        bev_gate_m=4.0,     # cross-camera 쌍 후보로 볼 GPS 거리(m)
        bev_lock_th=5.0,    # 누적점수 이 이상이면 동일객체 확정(락). 클수록 느리지만 확실
        bev_overlap_m=5.0,  # cam2↔cam3 겹침: 이 거리내면 즉시 동일차 통합(시간누적 없이)
):
    source = str(source)
    # cwd를 codes로 고정했으므로, 사용자가 준 상대경로 source는 '원래 실행 위치'(_ORIG_CWD)
    # 기준으로 복원. (URL/webcam 제외) → 어느 폴더에서 상대경로로 실행해도 파일을 찾음.
    if (not source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
            and not source.isnumeric() and not os.path.isabs(source)):
        _cand = os.path.join(_ORIG_CWD, source)
        if os.path.exists(_cand):
            source = _cand
    is_file = Path(source).suffix[1:] in VID_FORMATS
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    # Output directory
    if isinstance(yolo_weights, list):
        exp_name = Path(yolo_weights[0]).stem if len(yolo_weights) == 1 else 'ensemble'
    else:
        exp_name = yolo_weights.stem
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    if isinstance(yolo_weights, list):
        yolo_weights = yolo_weights[0]
    _use_trt = str(yolo_weights).endswith('.engine')
    trt_infer = None
    model = YOLO(str(yolo_weights), task='detect')
    if not _use_trt:
        model.to(device)

    stride = 32
    imgsz = check_img_size(imgsz, s=stride)
    names = {0: 'person', 1: 'cyclist', 2: 'car', 3: 'motorcycle', 4: 'truck'}

    # Dataloader
    # ── source → source_z(cam3), source_1(cam1) 자동 매핑 ──
    if 'cctv02' in source:
        source_z = source.replace('cctv02', 'cctv03')
        source_1 = source.replace('cctv02', 'cctv01')
    elif 'ch02' in source:
        source_z = source.replace('ch02', 'ch03')
        source_1 = source.replace('ch02', 'ch01')
    elif 'camera2' in source:
        source_z = source.replace('camera2', 'camera3')
        source_1 = source.replace('camera2', 'camera1')
    elif 'cam2' in source:
        source_z = source.replace('cam2', 'cam3')
        source_1 = source.replace('cam2', 'cam1')
    elif '.82' in source:
        source_z = source.replace('.82', '.83')
        source_1 = source.replace('.82', '.81')

    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = False  # 실증지 간헐 스파이크(track 급등) 해결 — 가변 배치 autotune 방지
        # RTSP 3캠: 각각 LoadStreams (소스 1개씩)
        dataset   = LoadStreams(source,   img_size=imgsz, stride=stride, auto=False)
        dataset_z = LoadStreams(source_z, img_size=imgsz, stride=stride, auto=False)
        dataset_1 = LoadStreams(source_1, img_size=imgsz, stride=stride, auto=False)
        nr_sources = 3
    else:
        dataset   = LoadImages(source,   img_size=imgsz, stride=stride, auto=False)
        dataset_z = LoadImages(source_z, img_size=imgsz, stride=stride, auto=False)
        dataset_1 = LoadImages(source_1, img_size=imgsz, stride=stride, auto=False)
        nr_sources = 3

    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Trackers: i=0 camera3(ocsort), i=1 camera2(strongsort_rotate), i=2 camera1(ocsort)
    tracker_list = []
    for i in range(nr_sources):
        method = 'strongsort_rotate' if i == 1 else 'ocsort'
        tracker = create_tracker(method, reid_weights, device, half)
        tracker_list.append(tracker)
        if hasattr(tracker_list[i], 'model') and hasattr(tracker_list[i].model, 'warmup'):
            tracker_list[i].model.warmup()

    outputs = [None] * nr_sources

    # Lane / speed state
    lane_1, lane_2, lane_3 = [], [], []
    mask = []
    speed_lane_1, speed_lane_2, speed_lane_3 = [], [], []
    traffic_lane_1, traffic_lane_2, traffic_lane_3 = [], [], []
    avg_speed_lane_1, avg_speed_lane_2, avg_speed_lane_3 = -999, -999, -999
    avg_speed_lane = [-999, -999, -999]

    # JSON output → outputs/ 에 생성 (지도 HTML이 /outputs/ 로 fetch. HV2X 루트서 http.server)
    _OUT = WEIGHTS.parent / 'outputs'
    _OUT.mkdir(exist_ok=True)
    cctv_traffic_json     = str(_OUT / "cctv-traffic.json")
    cctv_traffic_vds_json = str(_OUT / "cctv-traffic-vds.json")
    # Replay log: 매 frame의 JSON 스냅샷을 누적 (영상시간 기준 재생용)
    cctv_traffic_log_file = str(_OUT / "cctv-traffic-log.jsonl")
    # 새 실행 시 로그 초기화
    if Path(cctv_traffic_log_file).exists():
        Path(cctv_traffic_log_file).unlink()
    cctv_traffic     = []
    cctv_traffic_vds = []

    frame_index = 0
    REID_INTERVAL = 10  # N프레임마다 Re-ID(추출은 후보 crop 수 N에 비례). 사이 프레임은 트래커 ID로 유지 → 추출 빈도 1/10
    MA_WINDOW = 30
    _ma_total   = deque(maxlen=MA_WINDOW)
    _ma_infer   = deque(maxlen=MA_WINDOW)
    _ma_track   = deque(maxlen=MA_WINDOW)
    _ma_post    = deque(maxlen=MA_WINDOW)
    ch01_exist = False
    ch02_exist = False
    up_line, down_line, vup_line, vdown_line, mask, lane_1, lane_2, lane_3 = ch_line_rotate(0, 0)

    # Camera mapping: i=0→CAM203, i=1→CAM202, i=2→CAM101
    cctv_num = {0: 3, 1: 2, 2: 1}

    # cam203(zoom) ROI: cam203 시야가 cam202 커버영역 안에 겹쳐 동일 객체가
    # 두 카메라에서 중복 표출됨. cam203 커버영역(homography로 산출)을 ROI로 두고,
    # 그 안에 들어오는 cam202 검출은 drop → 겹침영역은 zoom인 cam203만 신뢰.
    CAM203_ROI = compute_camera_roi(toWGS_ch3)
    # ROI bounding box(AABB) 미리 계산 — 매 프레임 point_in_poly 전에 O(1) 사전배제용
    if CAM203_ROI:
        _roi_lats = [p[0] for p in CAM203_ROI]; _roi_lons = [p[1] for p in CAM203_ROI]
        CAM203_ROI_BBOX = (min(_roi_lats), max(_roi_lats), min(_roi_lons), max(_roi_lons))
    else:
        CAM203_ROI_BBOX = None
    print(f"[ROI] cam203 coverage hull: {len(CAM203_ROI)} verts "
          f"→ 이 영역 내 cam202 검출은 cam203 우선으로 drop")

    # GPS / speed tracking state
    prev_gps_ch3 = {}
    prev_gps_ch2 = {}
    prev_gps_ch1 = {}
    smooth_data  = {}
    # 정지 차량 display lock: (cam_idx, track_id) → {lat, lon, heading}
    # 정지 시 GPS noise/heading 무작위 회전으로 인한 박스 흔들림 방지
    # 내부 prev_gps 추적은 원본 GPS 그대로 (속도 계산 정확성 유지)
    disp_lock = {}
    # 연속 정지 프레임 카운터: (cam_idx, id) → count
    # 가시영역 이탈 시 1~2프레임의 일시적 speed=0이 잘못 lock 트리거하는 것 방지
    stationary_cnt = {}
    STATIONARY_LOCK_FRAMES = 30  # ≈1초 (30fps 기준) 연속 정지 시에만 lock 발동
    # 카메라별 도로 방향 (heading 무효 시 default — 정차차량 사각형을 도로와 평행하게)
    # i=0 cam3 / i=1 cam2 / i=2 cam1
    # cam2는 N-S 도로 (사용자 명시) → -π/2 (남쪽) 하드코딩
    # cam1, cam3는 homography 기반 추정 (이미지 중심선 → GPS)
    ROAD_HEADING = {
        0: compute_road_heading(toWGS_ch3),
        1: -math.pi / 2,  # cam2 N-S
        2: compute_road_heading(toWGS_ch1),
    }
    print(f"[ROAD_HEADING] cam3={ROAD_HEADING[0]:.3f}rad ({math.degrees(ROAD_HEADING[0]):.1f}°), "
          f"cam2={ROAD_HEADING[1]:.3f}rad ({math.degrees(ROAD_HEADING[1]):.1f}°), "
          f"cam1={ROAD_HEADING[2]:.3f}rad ({math.degrees(ROAD_HEADING[2]):.1f}°)")
    # Track별 class 평활화: (cam_idx, track_id) → deque(recent classes)
    # 한 프레임만 다른 클래스로 튀어도 majority vote로 일관 유지
    # + sticky class: 한 번 정해진 class는 명확한 majority(60%+)일 때만 변경
    cls_history  = {}
    cls_stable   = {}  # (cam_idx, track_id) → 마지막 안정 class
    CLS_SMOOTH_WIN = 11
    CLS_SWITCH_RATIO = 0.6  # 새 class로 바뀌려면 window 내 60%+ majority 필요
    last_result  = {}
    stop_start        = {}
    stop_moving_frames = {}
    STOP_RESET_FRAMES = 8
    SMOOTH_LEN    = 6
    VIDEO_FPS     = 30.0
    MIN_FRAME_GAP = 5

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    ai_latency = 0.0


    # AV(CAV) Re-ID state — CLIP+VeRi multi-query max similarity 기반
    av_car_flag = False
    AV_flag     = False
    eta_flag    = True
    AV_eta      = 0
    av_car_id     = -1
    av_car_id_ch2 = -1
    av_car_id_ch1 = -1
    av_best_sim_ch3 = 0.0
    av_best_sim_ch2 = 0.0
    av_best_sim_ch1 = 0.0
    # CAV 락 grace: 락된 트랙이 화면 끝에서 잘려 sim이 잠깐 떨어지거나 후보에서
    # 빠져도, 트랙이 살아있다고 보고 N프레임 동안 AV 식별을 유지(플리커 방지).
    av_miss_ch3 = 0
    av_miss_ch2 = 0
    av_miss_ch1 = 0
    CAV_GRACE = 30  # 약 1초 (30fps) — 이 동안 sim 미달이어도 AV 락 유지
    CAV_GLOBAL_ID  = int(cav_id)
    # 공유 extractor 먼저 생성 후 clip_weight 지정(쿼리 로드 전이어야 쿼리·크롭 비중 일치) → CAV·CV 공유
    _shared_ext = ReIDExtractor(weights=None, device=device)
    if reid_clip_weight is not None:
        _shared_ext.clip_weight = float(reid_clip_weight)
        print(f"[ReID] CLIP:색상 비중 override = {float(reid_clip_weight):.2f} : {1-float(reid_clip_weight):.2f}")
    cav_reid = CAVReID(cav_query, threshold=float(cav_threshold), device=device, extractor=_shared_ext)

    # CV(연결차) Re-ID state — CAV와 대칭 구조. --cv-reid 플래그일 때만 활성.
    cv_enabled = bool(cv_reid)
    CV_GLOBAL_ID = int(cv_id)
    cv_car_id = cv_car_id_ch2 = cv_car_id_ch1 = -1
    cv_best_sim_ch3 = cv_best_sim_ch2 = cv_best_sim_ch1 = 0.0
    cv_miss_ch3 = cv_miss_ch2 = cv_miss_ch1 = 0
    cv_car_flag = False
    cv_reid_engine = None
    if cv_enabled:
        # extractor 공유(cav_reid.extractor) → CLIP 중복 로드 방지
        cv_reid_engine = CAVReID(cv_query, threshold=float(cv_threshold), device=device, extractor=cav_reid.extractor)
        print(f"[CVReID] 활성화 — id={CV_GLOBAL_ID}, threshold={float(cv_threshold)}")

    # Real-time mode: 영상 파일을 RTSP처럼 wall time에 맞춰 frame drop
    realtime_start = time.time() if realtime else None
    if realtime:
        print(f"[REALTIME] 영상 파일을 wall clock에 맞춰 처리 (뒤처진 frame 자동 skip, VIDEO_FPS={VIDEO_FPS})")

    # 주행경로 기반 cross-camera ReID — 켜면 GPS 궤적 dedup+앵커로 cam2↔cam3 겹침 처리
    bev_tracker = (CrossCameraLinker(fps=VIDEO_FPS, gate_m=bev_gate_m, lock_th=bev_lock_th,
                                     overlap_dedup_m=bev_overlap_m)
                   if bev_reid else None)
    if bev_reid:
        print(f"[BEV-ReID] 주행경로 기반 cross-camera ReID 활성화 "
              f"(ID쌍 시간누적+락, gate={bev_gate_m}m, lock_th={bev_lock_th})")

    ##########정윤성
    V2X_IP = "0.0.0.0"
    V2X_PORT = 9000
    tcp_server = YoloTCPServer(host=V2X_IP, port=V2X_PORT)

    print("\n[INFO] AI Initialized. Waiting for C++ Client to connect before starting video...")
    tcp_server.client_connected.wait()
    print("[INFO] C++ Client connected! Starting real-time processing...\n")
    ##########

    for frame_idx, (full, zoom, ch1) in enumerate(zip(dataset, dataset_z, dataset_1)):
        # Frame skip: 영상 FPS에 추론 속도 못 맞출 때 N프레임마다 1번 처리
        # 처리 안 한 frame은 YOLO/tracker/JSON 업데이트 모두 skip
        if frame_skip > 1 and (frame_idx % frame_skip) != 0:
            continue
        # Real-time mode: wall time보다 영상 시간이 뒤처지면 skip해서 따라잡음
        if realtime:
            wall_elapsed = time.time() - realtime_start
            expected_frame = int(wall_elapsed * VIDEO_FPS)
            if frame_idx < expected_frame:
                continue  # 과거 frame은 skip (사용자가 보는 wall clock 시간에 맞춤)

        ##########정윤성
        if not tcp_server.client_connected.is_set():
            print("\n[INFO] Waiting for C++ Client to connect... (Video Paused)")
            tcp_server.client_connected.wait()
            print("[INFO] Client connected! Resuming video...\n")

        frame_console_logs = []
        ##########

        cctv_traffic.clear()
        cctv_traffic_vds.clear()

        merge_count = 0
        f_start = time.time()
        path,   im,   im0s,   vid_cap_f, s_f = full
        path_z, im_z, im0s_z, vid_cap_z, s_z = zoom
        path_1, im_1, im0s_1, vid_cap_1, s_1 = ch1

        # LoadStreams returns im0s as list, extract first element as numpy array
        if webcam:
            im0s   = np.ascontiguousarray(im0s[0])   if isinstance(im0s, list)   else im0s
            im0s_z = np.ascontiguousarray(im0s_z[0]) if isinstance(im0s_z, list) else im0s_z
            im0s_1 = np.ascontiguousarray(im0s_1[0]) if isinstance(im0s_1, list) else im0s_1
            path   = path[0]   if isinstance(path, list)   else path
            path_z = path_z[0] if isinstance(path_z, list) else path_z
            path_1 = path_1[0] if isinstance(path_1, list) else path_1

        # Inference
        t2 = time_sync()
        results = model.predict(
            source=[im0s, im0s_z, im0s_1],
            conf=conf_thres, iou=iou_thres,
            classes=classes, max_det=max_det,
            verbose=False, device=device, half=True
        )
        pred   = [yolo_to_det(results[0], device)]
        pred_z = [yolo_to_det(results[1], device)]
        pred_1 = [yolo_to_det(results[2], device)]

        t3 = time_sync()
        dt[1] += t3 - t2
        dt[2] += time_sync() - t3

        total_pred    = [pred_z, pred, pred_1]
        total_path    = [path_z, path, path_1]
        total_im      = [im_z, im, im_1]
        total_im0s    = [im0s_z, im0s, im0s_1]
        total_vid_cap = [vid_cap_z, vid_cap_f, vid_cap_1]
        total_s       = [s_z, s_f, s_1]
        save_pathes   = [0] * nr_sources
        reid_time = 0
        track_time = 0

        _t_prep_all=0; _t_track_all=0; _t_reid_all=0; _t_det_loop_all=0; _t_overlay_all=0; _t_savevid_all=0; _t_vds_all=0
        for i, det in enumerate(total_pred):
            det = det[0]
            seen += 1

            _t0 = time.time()
            if webcam:
                p, im0, _ = total_path[i], total_im0s[i].copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
                save_path = str(save_dir / (p.stem + str(i) + '.mp4'))
                txt_file_name = p.parent.name
            else:
                p, im0, _ = total_path[i], total_im0s[i].copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)
                else:
                    txt_file_name = p.parent.name
                    save_path = str(save_dir / p.parent.name)

            curr_frames[i] = im0
            save_pathes[i] = save_path
            s = total_s[i]
            vid_cap = total_vid_cap[i]
            txt_path = str(save_dir / 'tracks' / txt_file_name)
            s += '%gx%g ' % tuple(imgsz)
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=3, pil=not ascii, font_size=3)
            _t_prep_all += time.time() - _t0

            if det is not None and len(det):
                valid_mask = torch.isfinite(det).all(dim=1)
                det = det[valid_mask]
                if len(det) == 0:
                    det = None
            if det is not None and len(det):
                det[:, :4] = det[:, :4].round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # camera2: mask 내 detection 통과, 단 주차구역(parking_cam2) 안의 '차량류'만 제외
                # (사람/자전거는 주차구역 안이어도 통과 → 표출 유지)
                PARKING_EXCLUDE_CLS = {2, 3, 4}  # car, motorcycle, truck (person=0, cyclist=1은 유지)
                if i == 1 and len(det) > 0:
                    det_np = det.cpu().numpy()
                    road_dets = []
                    for d in det_np:
                        if not np.isfinite(d[:4]).all():
                            continue
                        foot = (int((d[0]+d[2])/2), int(d[3]))
                        if cv2.pointPolygonTest(mask, foot, False) < 0:
                            continue
                        if int(d[-1]) in PARKING_EXCLUDE_CLS and \
                           cv2.pointPolygonTest(parking_cam2, foot, False) >= 0:
                            continue  # 주차구역 안 차량류만 제외
                        road_dets.append(d)
                    det = (torch.tensor(np.array(road_dets), device=det.device, dtype=det.dtype)
                           if road_dets else
                           torch.zeros((0, det.shape[1]), device=det.device, dtype=det.dtype))

                t4 = time_sync()
                if i == 0:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                    zoom_av_candidate = []
                elif i == 1:
                    (outputs[i], lane_1, lane_2, lane_3,
                     speed_lane_1, speed_lane_2, speed_lane_3,
                     traffic_lane_1, traffic_lane_2, traffic_lane_3,
                     avg_speed_lane_1, avg_speed_lane_2, avg_speed_lane_3,
                     avg_speed_lane) = tracker_list[i].update(
                        det.cpu(), im0, lane_1, lane_2, lane_3,
                        speed_lane_1, speed_lane_2, speed_lane_3,
                        traffic_lane_1, traffic_lane_2, traffic_lane_3,
                        avg_speed_lane_1, avg_speed_lane_2, avg_speed_lane_3,
                        avg_speed_lane, mask, frame_idx)
                elif i == 2:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
                track_time += t5 - t4

                _t_reid_start = time.time()
                # ── camera3: zoom_av_candidate 빌드 (전체 outputs) ────────
                if i == 0:
                    zoom_img_crops = []
                    zoom_av_candidate = []
                    for output in outputs[i]:
                        bboxes = output[0:4]
                        zoom_id = output[4] * 100
                        zoom_img_crops.append(im0[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2])])
                        zoom_av_candidate.append(zoom_id)

                # ── CAV Re-ID (CLIP+VeRi, multi-query max similarity) ────────
                # Lock-and-verify: 한번 식별된 track_id가 outputs에 살아있고 sim>=threshold면 유지,
                # 잃거나 sim 떨어지면 top-1로 재검색 (sticky 오매칭과 per-frame flicker 모두 회피)
                if i == 0 and frame_idx % REID_INTERVAL == 0:
                    _H0, _W0 = im0.shape[0], im0.shape[1]
                    zoom_img_crops, zoom_av_candidate, zoom_edge = [], [], []
                    for output in outputs[i]:
                        bboxes = output[0:4]
                        zoom_id = output[4] * 100
                        crop = im0[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2])]
                        if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                            zoom_img_crops.append(crop); zoom_av_candidate.append(zoom_id)
                            zoom_edge.append(bboxes[0] <= 6 or bboxes[1] <= 6 or bboxes[2] >= _W0 - 6 or bboxes[3] >= _H0 - 6)
                    if not zoom_av_candidate:
                        if av_car_id != -1 and av_miss_ch3 < CAV_GRACE:
                            av_miss_ch3 += 1; av_car_flag = True   # 후보 전무(잠깐 사라짐) → grace
                        else:
                            av_car_id = -1; av_miss_ch3 = 0
                        if cv_enabled:
                            if cv_car_id != -1 and cv_miss_ch3 < CAV_GRACE:
                                cv_miss_ch3 += 1; cv_car_flag = True
                            else:
                                cv_car_id = -1; cv_miss_ch3 = 0
                    else:
                        _feats3 = cav_reid.extractor.extract_from_crops(zoom_img_crops)  # 추출 1회(CAV·CV 공유)
                        sims = cav_reid.best_sim_feats(_feats3)
                        locked_idx = zoom_av_candidate.index(av_car_id) if av_car_id != -1 and av_car_id in zoom_av_candidate else None
                        if locked_idx is not None and sims[locked_idx] >= cav_reid.threshold:
                            av_miss_ch3 = 0
                            av_best_sim_ch3 = float(sims[locked_idx]); av_car_flag = True
                        elif locked_idx is not None and zoom_edge[locked_idx] and av_miss_ch3 < CAV_GRACE:
                            av_miss_ch3 += 1; av_car_flag = True   # 화면끝 잘림 → grace (잘렸을 때만)
                        elif locked_idx is None and av_car_id != -1 and av_miss_ch3 < CAV_GRACE:
                            av_miss_ch3 += 1; av_car_flag = True   # 후보서 잠깐 빠짐 → grace
                        else:
                            # 완전노출 & sim 미달 → 즉시 해제(오인식 자가교정) 후 재탐색
                            av_car_id = -1; av_miss_ch3 = 0
                            bi = int(np.argmax(sims))
                            if sims[bi] >= cav_reid.threshold:
                                av_car_id = zoom_av_candidate[bi]; av_miss_ch3 = 0; av_car_flag = True
                                av_best_sim_ch3 = float(sims[bi])
                                print(f"[CAVReID] camera3 matched, id={av_car_id}, sim={sims[bi]:.3f}")
                        # ── CV 매칭 (CAV와 대칭, 상호배타: CAV가 가진 crop 제외) ──
                        if cv_enabled:
                            cv_sims = cv_reid_engine.best_sim_feats(_feats3)
                            _cav_own = zoom_av_candidate.index(av_car_id) if (av_car_id != -1 and av_car_id in zoom_av_candidate) else None
                            if _cav_own is not None and _cav_own < len(cv_sims):
                                cv_sims[_cav_own] = -1.0   # CAV로 배정된 crop은 CV 후보서 제외
                            cvl = zoom_av_candidate.index(cv_car_id) if cv_car_id != -1 and cv_car_id in zoom_av_candidate else None
                            if cvl is not None and cv_sims[cvl] >= cv_reid_engine.threshold:
                                cv_miss_ch3 = 0; cv_best_sim_ch3 = float(cv_sims[cvl]); cv_car_flag = True
                            elif cvl is not None and zoom_edge[cvl] and cv_miss_ch3 < CAV_GRACE:
                                cv_miss_ch3 += 1; cv_car_flag = True
                            elif cvl is None and cv_car_id != -1 and cv_miss_ch3 < CAV_GRACE:
                                cv_miss_ch3 += 1; cv_car_flag = True
                            else:
                                cv_car_id = -1; cv_miss_ch3 = 0
                                if len(cv_sims):
                                    bj = int(np.argmax(cv_sims))
                                    if cv_sims[bj] >= cv_reid_engine.threshold:
                                        cv_car_id = zoom_av_candidate[bj]; cv_miss_ch3 = 0; cv_car_flag = True
                                        cv_best_sim_ch3 = float(cv_sims[bj])
                                        print(f"[CVReID] camera3 matched, id={cv_car_id}, sim={cv_sims[bj]:.3f}")

                if i == 1 and frame_idx % REID_INTERVAL == 0:
                    _H1, _W1 = im0.shape[0], im0.shape[1]
                    crops, cids, cedge = [], [], []
                    if len(outputs[i]) > 0:
                        for output in outputs[i]:
                            bboxes = output[0:4]
                            bbox_cx = (bboxes[2] + bboxes[0]) / 2
                            bbox_cy = (bboxes[3] + bboxes[1]) / 2
                            if cv2.pointPolygonTest(lane_1, (float(bbox_cx), float(bbox_cy)), False) >= 0:
                                crop = im0[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2])]
                                if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                                    crops.append(crop); cids.append(output[4])
                                    cedge.append(bboxes[0] <= 6 or bboxes[1] <= 6 or bboxes[2] >= _W1 - 6 or bboxes[3] >= _H1 - 6)
                    if not cids:
                        if av_car_id_ch2 != -1 and av_miss_ch2 < CAV_GRACE:
                            av_miss_ch2 += 1; av_car_flag = True
                        else:
                            av_car_id_ch2 = -1; av_miss_ch2 = 0
                        if cv_enabled:
                            if cv_car_id_ch2 != -1 and cv_miss_ch2 < CAV_GRACE:
                                cv_miss_ch2 += 1; cv_car_flag = True
                            else:
                                cv_car_id_ch2 = -1; cv_miss_ch2 = 0
                    else:
                        _feats2 = cav_reid.extractor.extract_from_crops(crops)  # 추출 1회(CAV·CV 공유)
                        sims = cav_reid.best_sim_feats(_feats2)
                        locked_idx = cids.index(av_car_id_ch2) if av_car_id_ch2 != -1 and av_car_id_ch2 in cids else None
                        if locked_idx is not None and sims[locked_idx] >= cav_reid.threshold:
                            av_miss_ch2 = 0
                            av_best_sim_ch2 = float(sims[locked_idx]); av_car_flag = True
                        elif locked_idx is not None and cedge[locked_idx] and av_miss_ch2 < CAV_GRACE:
                            av_miss_ch2 += 1; av_car_flag = True   # 화면끝 잘림 → grace
                        elif locked_idx is None and av_car_id_ch2 != -1 and av_miss_ch2 < CAV_GRACE:
                            av_miss_ch2 += 1; av_car_flag = True   # 후보서 잠깐 빠짐 → grace
                        else:
                            av_car_id_ch2 = -1; av_miss_ch2 = 0
                            bi = int(np.argmax(sims))
                            if sims[bi] >= cav_reid.threshold:
                                av_car_id_ch2 = cids[bi]; av_miss_ch2 = 0; av_car_flag = True
                                av_best_sim_ch2 = float(sims[bi])
                                print(f"[CAVReID] camera2 matched, id={av_car_id_ch2}, sim={sims[bi]:.3f}")
                        # ── CV 매칭 (상호배타: CAV가 가진 crop 제외) ──
                        if cv_enabled:
                            cv_sims = cv_reid_engine.best_sim_feats(_feats2)
                            _cav_own = cids.index(av_car_id_ch2) if (av_car_id_ch2 != -1 and av_car_id_ch2 in cids) else None
                            if _cav_own is not None and _cav_own < len(cv_sims):
                                cv_sims[_cav_own] = -1.0
                            cvl = cids.index(cv_car_id_ch2) if cv_car_id_ch2 != -1 and cv_car_id_ch2 in cids else None
                            if cvl is not None and cv_sims[cvl] >= cv_reid_engine.threshold:
                                cv_miss_ch2 = 0; cv_best_sim_ch2 = float(cv_sims[cvl]); cv_car_flag = True
                            elif cvl is not None and cedge[cvl] and cv_miss_ch2 < CAV_GRACE:
                                cv_miss_ch2 += 1; cv_car_flag = True
                            elif cvl is None and cv_car_id_ch2 != -1 and cv_miss_ch2 < CAV_GRACE:
                                cv_miss_ch2 += 1; cv_car_flag = True
                            else:
                                cv_car_id_ch2 = -1; cv_miss_ch2 = 0
                                if len(cv_sims):
                                    bj = int(np.argmax(cv_sims))
                                    if cv_sims[bj] >= cv_reid_engine.threshold:
                                        cv_car_id_ch2 = cids[bj]; cv_miss_ch2 = 0; cv_car_flag = True
                                        cv_best_sim_ch2 = float(cv_sims[bj])
                                        print(f"[CVReID] camera2 matched, id={cv_car_id_ch2}, sim={cv_sims[bj]:.3f}")

                if i == 2 and frame_idx % REID_INTERVAL == 0:
                    _H2, _W2 = im0.shape[0], im0.shape[1]
                    crops, cids, cedge = [], [], []
                    if len(outputs[i]) > 0:
                        for output in outputs[i]:
                            bboxes = output[0:4]
                            bbox_cx = (bboxes[2] + bboxes[0]) / 2
                            bbox_cy = (bboxes[3] + bboxes[1]) / 2
                            in_lane = any(cv2.pointPolygonTest(lp, (float(bbox_cx), float(bbox_cy)), False) >= 0
                                          for lp in cam1_lanes)
                            if in_lane:
                                crop = im0[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2])]
                                if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                                    crops.append(crop); cids.append(output[4])
                                    cedge.append(bboxes[0] <= 6 or bboxes[1] <= 6 or bboxes[2] >= _W2 - 6 or bboxes[3] >= _H2 - 6)
                    if not cids:
                        if av_car_id_ch1 != -1 and av_miss_ch1 < CAV_GRACE:
                            av_miss_ch1 += 1; av_car_flag = True
                        else:
                            av_car_id_ch1 = -1; av_miss_ch1 = 0
                        if cv_enabled:
                            if cv_car_id_ch1 != -1 and cv_miss_ch1 < CAV_GRACE:
                                cv_miss_ch1 += 1; cv_car_flag = True
                            else:
                                cv_car_id_ch1 = -1; cv_miss_ch1 = 0
                    else:
                        _feats1 = cav_reid.extractor.extract_from_crops(crops)  # 추출 1회(CAV·CV 공유)
                        sims = cav_reid.best_sim_feats(_feats1)
                        locked_idx = cids.index(av_car_id_ch1) if av_car_id_ch1 != -1 and av_car_id_ch1 in cids else None
                        if locked_idx is not None and sims[locked_idx] >= cav_reid.threshold:
                            av_miss_ch1 = 0
                            av_best_sim_ch1 = float(sims[locked_idx]); av_car_flag = True
                        elif locked_idx is not None and cedge[locked_idx] and av_miss_ch1 < CAV_GRACE:
                            av_miss_ch1 += 1; av_car_flag = True   # 화면끝 잘림 → grace
                        elif locked_idx is None and av_car_id_ch1 != -1 and av_miss_ch1 < CAV_GRACE:
                            av_miss_ch1 += 1; av_car_flag = True   # 후보서 잠깐 빠짐 → grace
                        else:
                            av_car_id_ch1 = -1; av_miss_ch1 = 0
                            bi = int(np.argmax(sims))
                            if sims[bi] >= cav_reid.threshold:
                                av_car_id_ch1 = cids[bi]; av_miss_ch1 = 0; av_car_flag = True
                                av_best_sim_ch1 = float(sims[bi])
                                print(f"[CAVReID] camera1 matched, id={av_car_id_ch1}, sim={sims[bi]:.3f}")
                        # ── CV 매칭 (상호배타: CAV가 가진 crop 제외) ──
                        if cv_enabled:
                            cv_sims = cv_reid_engine.best_sim_feats(_feats1)
                            _cav_own = cids.index(av_car_id_ch1) if (av_car_id_ch1 != -1 and av_car_id_ch1 in cids) else None
                            if _cav_own is not None and _cav_own < len(cv_sims):
                                cv_sims[_cav_own] = -1.0
                            cvl = cids.index(cv_car_id_ch1) if cv_car_id_ch1 != -1 and cv_car_id_ch1 in cids else None
                            if cvl is not None and cv_sims[cvl] >= cv_reid_engine.threshold:
                                cv_miss_ch1 = 0; cv_best_sim_ch1 = float(cv_sims[cvl]); cv_car_flag = True
                            elif cvl is not None and cedge[cvl] and cv_miss_ch1 < CAV_GRACE:
                                cv_miss_ch1 += 1; cv_car_flag = True
                            elif cvl is None and cv_car_id_ch1 != -1 and cv_miss_ch1 < CAV_GRACE:
                                cv_miss_ch1 += 1; cv_car_flag = True
                            else:
                                cv_car_id_ch1 = -1; cv_miss_ch1 = 0
                                if len(cv_sims):
                                    bj = int(np.argmax(cv_sims))
                                    if cv_sims[bj] >= cv_reid_engine.threshold:
                                        cv_car_id_ch1 = cids[bj]; cv_miss_ch1 = 0; cv_car_flag = True
                                        cv_best_sim_ch1 = float(cv_sims[bj])
                                        print(f"[CVReID] camera1 matched, id={cv_car_id_ch1}, sim={cv_sims[bj]:.3f}")

                # camera3 검출 수 (JSON merge_count 필드용)
                # ※ 구 cam3→cam2 픽셀-homography id 매칭(2022 npy 기반)은 제거됨.
                #    cam2↔cam3 겹침 dedup은 --bev-reid(주행경로 linker)가 담당.
                if i == 0:
                    merge_count = len(outputs[i])

                _t_reid_all += time.time() - _t_reid_start

                if outputs[1] is not None:
                    ch02_exist = len(outputs[1]) > 0
                if outputs[2] is not None:
                    ch01_exist = len(outputs[2]) > 0

                _t_det_start = time.time()
                if len(outputs[i]) > 0:
                    AV_flag = False
                    for j, output in enumerate(outputs[i]):
                        bboxes = output[0:4]
                        id  = output[4]
                        cls = output[5]

                        # Class 평활화 (track별 최근 N프레임 majority + sticky)
                        # - 신규 track: window 채워질 때까진 현재 majority 사용
                        # - 안정 track: 60%+ majority일 때만 새 class로 변경 (그 외엔 마지막 stable 유지)
                        _ckey = (i, int(id))
                        _hist = cls_history.get(_ckey)
                        if _hist is None:
                            _hist = deque(maxlen=CLS_SMOOTH_WIN)
                            cls_history[_ckey] = _hist
                        _hist.append(int(cls))
                        _counts = {c: list(_hist).count(c) for c in set(_hist)}
                        _candidate = max(_counts, key=_counts.get)
                        _threshold = max(1, int(len(_hist) * CLS_SWITCH_RATIO))
                        if _counts[_candidate] >= _threshold:
                            cls_stable[_ckey] = _candidate
                        cls = cls_stable.get(_ckey, _candidate)

                        if i == 1:
                            final_speed = output[7]
                            bbox_x = output[8]
                            bbox_y = output[9]

                        # ID adjustment (CAV로 식별된 track은 모든 카메라에서 CAV_GLOBAL_ID로 통일)
                        if i == 0 and id != CAV_GLOBAL_ID:
                            id = id * 100
                            if av_car_flag and id == av_car_id:
                                id = CAV_GLOBAL_ID
                            elif cv_enabled and cv_car_flag and id == cv_car_id:
                                id = CV_GLOBAL_ID
                        if i == 1 and id == av_car_id_ch2:
                            id = CAV_GLOBAL_ID; AV_flag = True
                        elif i == 1 and cv_enabled and id == cv_car_id_ch2:
                            id = CV_GLOBAL_ID
                        if i == 2 and id == av_car_id_ch1:
                            id = CAV_GLOBAL_ID; AV_flag = True
                        elif i == 2 and cv_enabled and id == cv_car_id_ch1:
                            id = CV_GLOBAL_ID

                        # ── GPS + speed + stop (camera2) ──────────────────
                        if len(output) > 8 and i == 1:
                            gps = toWGS(output[8], output[9])
                            lat_now, lon_now = gps[0], gps[1]
                            speed_val, heading_val = 999, 999
                            if id in prev_gps_ch2:
                                prev_lat, prev_lon, prev_fidx = prev_gps_ch2[id]
                                frame_gap = frame_idx - prev_fidx
                                if frame_gap >= MIN_FRAME_GAP:
                                    h_raw, s_raw = calc_heading_speed(lat_now, lon_now, prev_lat, prev_lon, frame_gap, VIDEO_FPS)
                                    heading_val, speed_val = smooth_heading_speed(id, h_raw, s_raw, smooth_data, SMOOTH_LEN)
                                    last_result[id] = (heading_val, speed_val)
                                    prev_gps_ch2[id] = (lat_now, lon_now, frame_idx)
                                elif id in last_result:
                                    heading_val, speed_val = last_result[id]
                            else:
                                prev_gps_ch2[id] = (lat_now, lon_now, frame_idx)
                            stop_duration = _update_stop(id, 1, speed_val, stop_start, stop_moving_frames, STOP_RESET_FRAMES, SPEED_MIN_KMH, (None if webcam else frame_idx), VIDEO_FPS)
                            ped_zone = get_ped_zone((output[0]+output[2])/2, output[3],
                                                    crosswalk_cam2, sidewalk_cam2, road_cam2) if int(cls) == 0 else None
                            # 정지 시 display 좌표/heading 고정 (사각형 흔들림 방지)
                            # 연속 N프레임 정지 확인 후 lock — 가시영역 이탈 시 false-lock 방지
                            _dkey = (1, id)
                            if speed_val != 999 and speed_val < SPEED_MIN_KMH:
                                stationary_cnt[_dkey] = stationary_cnt.get(_dkey, 0) + 1
                                if stationary_cnt[_dkey] >= STATIONARY_LOCK_FRAMES:
                                    # GPS 점프 시(>20m) lock 갱신 (track 재할당 의심)
                                    if _dkey in disp_lock:
                                        _d_lat = lat_now - disp_lock[_dkey]['lat']
                                        _d_lon = lon_now - disp_lock[_dkey]['lon']
                                        _d_m = math.sqrt((_d_lat*111320)**2 + (_d_lon*111320*0.81)**2)
                                        if _d_m > 20:
                                            disp_lock.pop(_dkey)
                                    if _dkey not in disp_lock:
                                        disp_lock[_dkey] = {'lat': lat_now, 'lon': lon_now,
                                                            'heading': ROAD_HEADING[i]}
                                    lat_now = disp_lock[_dkey]['lat']
                                    lon_now = disp_lock[_dkey]['lon']
                                    heading_val = disp_lock[_dkey]['heading']
                            else:
                                stationary_cnt.pop(_dkey, None)
                                disp_lock.pop(_dkey, None)
                            # cam2·cam3 FOV 가장자리(프레임 경계 6px 이내)에선 발끝점 클리핑+homography 외삽으로
                            # GPS·heading이 불안정 → 방향 박스가 크게 틀어짐. 위치(점)는 유지하되 박스는 생략(None).
                            # (커브 진입구역이라 ROAD_HEADING 스냅도 부적합) cam1(i==2)은 제외.
                            bbox_gps = gps_oriented_rect(lat_now, lon_now, heading_val if heading_val != 999 else ROAD_HEADING[i], cls)
                            cctv_traffic.append({"cctv": cctv_num[i], "class": int(cls), "id": id,
                                                 "bbox": list(output[0:4]), "GPS": [lat_now, lon_now],
                                                 "bbox_gps": bbox_gps,
                                                 "speed": speed_val, "heading": heading_val,
                                                 "lane": -1, "stop_duration": stop_duration,
                                                 "lane_occupancy": [-1, -1],
                                                 "ped_zone": ped_zone if ped_zone is not None else "none"})

                        # ── GPS + speed + stop (camera3) ──────────────────
                        if i == 0:
                            # bbox 하단중앙(지면 접촉점) 기준 — cam1/cam2와 통일
                            # (기존 (x1,y1) 좌상단은 지면이 아니라 homography 투영이 어긋났음)
                            gps = toWGS_ch3((output[0] + output[2]) / 2, (output[1] + output[3]) / 2)  # [실험] foot→box 중심
                            lat_now, lon_now = gps[0], gps[1]
                            # cam3 heading/속도는 박스 중앙(x_center, (y1+y2)/2) 기준.
                            #   접지점(y2)은 엣지에서 잘려 방향이 튀고, 상단(y1)은 지붕 높이라 왜곡 큼(A/B 67°).
                            #   박스 중앙이 절충(A/B 32°) + 위치(박스중앙)와 일관.
                            _gps_h = toWGS_ch3((output[0] + output[2]) / 2, (output[1] + output[3]) / 2)
                            lat_h, lon_h = _gps_h[0], _gps_h[1]
                            speed_val, heading_val = 999, 999
                            if id in prev_gps_ch3:
                                prev_lat, prev_lon, prev_fidx = prev_gps_ch3[id]
                                frame_gap = frame_idx - prev_fidx
                                if frame_gap >= MIN_FRAME_GAP:
                                    h_raw, s_raw = calc_heading_speed(lat_h, lon_h, prev_lat, prev_lon, frame_gap, VIDEO_FPS)
                                    heading_val, speed_val = smooth_heading_speed(id, h_raw, s_raw, smooth_data, SMOOTH_LEN)
                                    last_result[id] = (heading_val, speed_val)
                                    prev_gps_ch3[id] = (lat_h, lon_h, frame_idx)
                                elif id in last_result:
                                    heading_val, speed_val = last_result[id]
                            else:
                                prev_gps_ch3[id] = (lat_h, lon_h, frame_idx)
                            _update_stop(id, 0, speed_val, stop_start, stop_moving_frames, STOP_RESET_FRAMES, SPEED_MIN_KMH, (None if webcam else frame_idx), VIDEO_FPS)
                            stop_duration = -1  # camera3 stop detection disabled → -1(측정 안 함)
                            ped_zone = get_ped_zone((output[0]+output[2])/2, output[3],
                                                    crosswalk_cam3, sidewalk_cam3, road_cam3) if int(cls) == 0 else None
                            # 정지 시 display 좌표/heading 고정
                            _dkey = (0, id)
                            if speed_val != 999 and speed_val < SPEED_MIN_KMH:
                                stationary_cnt[_dkey] = stationary_cnt.get(_dkey, 0) + 1
                                if stationary_cnt[_dkey] >= STATIONARY_LOCK_FRAMES:
                                    if _dkey in disp_lock:
                                        _d_lat = lat_now - disp_lock[_dkey]['lat']
                                        _d_lon = lon_now - disp_lock[_dkey]['lon']
                                        _d_m = math.sqrt((_d_lat*111320)**2 + (_d_lon*111320*0.81)**2)
                                        if _d_m > 20:
                                            disp_lock.pop(_dkey)
                                    if _dkey not in disp_lock:
                                        disp_lock[_dkey] = {'lat': lat_now, 'lon': lon_now,
                                                            'heading': ROAD_HEADING[i]}
                                    lat_now = disp_lock[_dkey]['lat']
                                    lon_now = disp_lock[_dkey]['lon']
                                    heading_val = disp_lock[_dkey]['heading']
                            else:
                                stationary_cnt.pop(_dkey, None)
                                disp_lock.pop(_dkey, None)
                            # cam2·cam3 FOV 가장자리(프레임 경계 6px 이내)에선 발끝점 클리핑+homography 외삽으로
                            # GPS·heading이 불안정 → 방향 박스가 크게 틀어짐. 위치(점)는 유지하되 박스는 생략(None).
                            # (커브 진입구역이라 ROAD_HEADING 스냅도 부적합) cam1(i==2)은 제외.
                            bbox_gps = gps_oriented_rect(lat_now, lon_now, heading_val if heading_val != 999 else ROAD_HEADING[i], cls)
                            cctv_traffic.append({"cctv": cctv_num[i], "class": int(cls), "id": id,
                                                 "bbox": list(output[0:4]), "GPS": [lat_now, lon_now],
                                                 "bbox_gps": bbox_gps,
                                                 "speed": speed_val, "heading": heading_val,
                                                 "lane": -1, "stop_duration": stop_duration,
                                                 "lane_occupancy": [-1, -1],
                                                 "ped_zone": ped_zone if ped_zone is not None else "none"})

                        # ── GPS + speed + stop (camera1) ──────────────────
                        if i == 2:
                            gps = toWGS_ch1((output[0]+output[2])/2, (output[1]+output[3])/2)  # [실험] foot→box 중심
                            lat_now, lon_now = gps[0], gps[1]
                            speed_val, heading_val = 999, 999
                            if id in prev_gps_ch1:
                                prev_lat, prev_lon, prev_fidx = prev_gps_ch1[id]
                                frame_gap = frame_idx - prev_fidx
                                if frame_gap >= MIN_FRAME_GAP:
                                    h_raw, s_raw = calc_heading_speed(lat_now, lon_now, prev_lat, prev_lon, frame_gap, VIDEO_FPS)
                                    heading_val, speed_val = smooth_heading_speed(id, h_raw, s_raw, smooth_data, SMOOTH_LEN)
                                    last_result[id] = (heading_val, speed_val)
                                    prev_gps_ch1[id] = (lat_now, lon_now, frame_idx)
                                elif id in last_result:
                                    heading_val, speed_val = last_result[id]
                            else:
                                prev_gps_ch1[id] = (lat_now, lon_now, frame_idx)
                            stop_duration = _update_stop(id, 2, speed_val, stop_start, stop_moving_frames, STOP_RESET_FRAMES, SPEED_MIN_KMH, (None if webcam else frame_idx), VIDEO_FPS)
                            lane_idx, lane_occ = (None, [-1, -1])   # 차선별 점유율 [lane1, lane2], 미정차=[-1,-1]
                            if stop_duration > 0:
                                lane_idx, lane_occ = calc_lane_occupancy(output[0:4], cam1_lanes)
                            ped_zone = (get_ped_zone((output[0]+output[2])/2, output[3],
                                                     crosswalk_cam1, sidewalk_cam1, road_cam1)
                                        if int(cls) == 0 and sidewalk_cam1 is not None else None)
                            # 정지 시 display 좌표/heading 고정
                            _dkey = (2, id)
                            if speed_val != 999 and speed_val < SPEED_MIN_KMH:
                                stationary_cnt[_dkey] = stationary_cnt.get(_dkey, 0) + 1
                                if stationary_cnt[_dkey] >= STATIONARY_LOCK_FRAMES:
                                    if _dkey in disp_lock:
                                        _d_lat = lat_now - disp_lock[_dkey]['lat']
                                        _d_lon = lon_now - disp_lock[_dkey]['lon']
                                        _d_m = math.sqrt((_d_lat*111320)**2 + (_d_lon*111320*0.81)**2)
                                        if _d_m > 20:
                                            disp_lock.pop(_dkey)
                                    if _dkey not in disp_lock:
                                        disp_lock[_dkey] = {'lat': lat_now, 'lon': lon_now,
                                                            'heading': ROAD_HEADING[i]}
                                    lat_now = disp_lock[_dkey]['lat']
                                    lon_now = disp_lock[_dkey]['lon']
                                    heading_val = disp_lock[_dkey]['heading']
                            else:
                                stationary_cnt.pop(_dkey, None)
                                disp_lock.pop(_dkey, None)
                            # cam2·cam3 FOV 가장자리(프레임 경계 6px 이내)에선 발끝점 클리핑+homography 외삽으로
                            # GPS·heading이 불안정 → 방향 박스가 크게 틀어짐. 위치(점)는 유지하되 박스는 생략(None).
                            # (커브 진입구역이라 ROAD_HEADING 스냅도 부적합) cam1(i==2)은 제외.
                            bbox_gps = gps_oriented_rect(lat_now, lon_now, heading_val if heading_val != 999 else ROAD_HEADING[i], cls)
                            cctv_traffic.append({"cctv": cctv_num[i], "class": int(cls), "id": id,
                                                 "bbox": list(output[0:4]), "GPS": [lat_now, lon_now],
                                                 "bbox_gps": bbox_gps,
                                                 "speed": speed_val, "heading": heading_val,
                                                 "lane": lane_idx if lane_idx is not None else -1,
                                                 "stop_duration": stop_duration,
                                                 "lane_occupancy": lane_occ,
                                                 "ped_zone": ped_zone if ped_zone is not None else "none"})

                        if save_txt:
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (
                                    frame_idx + 1, id,
                                    output[0], output[1],
                                    output[2] - output[0], output[3] - output[1],
                                    -1, -1, -1, i))

                        # ── Draw bounding boxes ───────────────────────────
                        if i == 1 and output[3] > 80 and (save_vid or save_crop or show_vid):
                            c = int(cls); id = int(id)
                            disp_spd = int(abs(speed_val)) if speed_val != 999 else 0
                            label = f'{names[c]} {disp_spd}km/h'
                            if id == CAV_GLOBAL_ID:
                                c = 50
                                label = f'AV {disp_spd}km/h'
                                AV_flag = True
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(bboxes, imc,
                                             file=save_dir / 'crops' / names[c] / f'{id}' / f'{p.stem}.jpg',
                                             BGR=True)

                        elif i == 0 and (save_vid or save_crop or show_vid):
                            c = int(cls); id = int(id)
                            label = 'AV' if id == CAV_GLOBAL_ID else f'{names[c]}'
                            if id == CAV_GLOBAL_ID:
                                c = 50
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(bboxes, imc,
                                             file=save_dir / 'crops' / names[c] / f'{id}' / f'{p.stem}.jpg',
                                             BGR=True)

                        elif i == 2 and (save_vid or save_crop or show_vid):
                            id = int(id); c = int(cls)
                            if id == CAV_GLOBAL_ID:
                                c = 50
                            label = 'AV' if id == CAV_GLOBAL_ID else f'{names[c]}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(bboxes, imc,
                                             file=save_dir / 'crops' / names[c] / f'{id}' / f'{p.stem}.jpg',
                                             BGR=True)

                if not show_sdsm:
                    LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')
                ##########정윤성
                frame_console_logs.append(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')
                ##########
                _t_det_loop_all += time.time() - _t_det_start

            else:
                if not show_sdsm:
                    LOGGER.info('No detections')
                ##########정윤성
                frame_console_logs.append('No detections')
                ##########
                if i == 0:
                    zoom_av_candidate = []
                outputs[i] = []

            _t_overlay_start = time.time()
            # ── Overlay status text ───────────────────────────────────────
            im0 = annotator.result()

            if i == 0:
                label = "AV ID: Identified" if av_car_id != -1 else "AV ID: Not Identified"
                color = (255, 0, 0) if av_car_id != -1 else (255, 255, 255)
                cv2.putText(im0, label, (50, 80), 0, 1, color, 2)
                status_color = (0, 0, 255) if ch02_exist else (255, 255, 255)
                cv2.putText(im0, "Status: Warning" if ch02_exist else "Status: Safe",
                            (50, 40), 0, 1, status_color, 2)

            if i == 2:
                label = "AV ID: Identified" if av_car_id_ch1 != -1 else "AV ID: Not Identified"
                color = (255, 0, 0) if av_car_id_ch1 != -1 else (255, 255, 255)
                cv2.putText(im0, label, (50, 80), 0, 1, color, 2)

            if i == 1:
                if AV_flag:
                    cv2.putText(im0, "AV Re-ID: Re-Identified", (50, 80), 0, 1, (255, 255, 0), 2)
                else:
                    cv2.putText(im0, "AV Re-ID: Not Identified", (50, 80), 0, 1, (255, 255, 255), 2)
                status_color = (0, 0, 255) if ch01_exist else (255, 255, 255)
                cv2.putText(im0, "Status: Warning" if ch01_exist else "Status: Safe",
                            (50, 40), 0, 1, status_color, 2)

                lane1_speed, lane1_count = calc_lane_speed(speed_lane_1)
                lane2_speed, lane2_count = calc_lane_speed(speed_lane_2)
                lane3_speed, lane3_count = calc_lane_speed(speed_lane_3)
                cctv_traffic_vds.append({"cctv": cctv_num[i],
                                         "lane1_speed": lane1_speed, "lane1_count": lane1_count,
                                         "lane2_speed": lane2_speed, "lane2_count": lane2_count,
                                         "lane3_speed": lane3_speed, "lane3_count": lane3_count,
                                         "merge_count": int(merge_count), "timestamp": int(time.time())})
                with open(cctv_traffic_vds_json, "w", encoding='utf-8') as f:
                    json.dump(cctv_traffic_vds, f, indent=4)

                if frame_index % 150 == 0:
                    speed_lane_1.clear(); speed_lane_2.clear(); speed_lane_3.clear()
                    traffic_lane_1.clear(); traffic_lane_2.clear(); traffic_lane_3.clear()

            if i == 2:
                status_color = (0, 0, 255) if ch02_exist else (255, 255, 255)
                cv2.putText(im0, "Status: Warning" if ch02_exist else "Status: Safe",
                            (50, 40), 0, 1, status_color, 2)

            _t_overlay_all += time.time() - _t_overlay_start

            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            _t_savevid_start = time.time()
            if save_vid:
                if vid_path[i] != save_path:
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            _t_savevid_all += time.time() - _t_savevid_start
            prev_frames[i] = curr_frames[i]

        _t_json_start = time.time()
        ai_latency = round(time.time() - f_start, 4)
        _inf  = round((t3 - t2) * 1000, 1)
        _trk  = round(track_time * 1000, 1)
        _post = round((ai_latency * 1000) - _inf - _trk, 1)
        _fps  = round(1.0 / ai_latency, 1) if ai_latency > 0 else 0
        _ma_total.append(ai_latency * 1000)
        _ma_infer.append(_inf)
        _ma_track.append(_trk)
        _ma_post.append(_post)
        _ma_fps = round(1000.0 / (sum(_ma_total) / len(_ma_total)), 1) if _ma_total else 0
        if not show_sdsm:
            print(f"[TIMING #{frame_idx}] infer:{_inf}ms | track:{_trk}ms | post:{_post}ms | total:{ai_latency*1000:.1f}ms | FPS:{_fps}")
            print(f"  [MA-{MA_WINDOW}] infer:{sum(_ma_infer)/len(_ma_infer):.1f}ms | track:{sum(_ma_track)/len(_ma_track):.1f}ms | post:{sum(_ma_post)/len(_ma_post):.1f}ms | total:{sum(_ma_total)/len(_ma_total):.1f}ms | FPS:{_ma_fps}")
            print(f"  [POST] prep:{_t_prep_all*1000:.1f} | reid:{_t_reid_all*1000:.1f} | det_loop:{_t_det_loop_all*1000:.1f} | overlay:{_t_overlay_all*1000:.1f} | savevid:{_t_savevid_all*1000:.1f}ms")

        ##########정윤성
        frame_console_logs.extend([
            f"[TIMING #{frame_idx}] infer:{_inf}ms | track:{_trk}ms | post:{_post}ms | total:{ai_latency*1000:.1f}ms | FPS:{_fps}",
            f"  [MA-{MA_WINDOW}] infer:{sum(_ma_infer)/len(_ma_infer):.1f}ms | track:{sum(_ma_track)/len(_ma_track):.1f}ms | post:{sum(_ma_post)/len(_ma_post):.1f}ms | total:{sum(_ma_total)/len(_ma_total):.1f}ms | FPS:{_ma_fps}",
            f"  [POST] prep:{_t_prep_all*1000:.1f} | reid:{_t_reid_all*1000:.1f} | det_loop:{_t_det_loop_all*1000:.1f} | overlay:{_t_overlay_all*1000:.1f} | savevid:{_t_savevid_all*1000:.1f}ms"
        ])
        ##########

        # ── 겹침 dedup (presence 기반): cam2가 같이 보면 cam3 drop ──────────────
        # cam3 검출 근처(8m, 같은 차량그룹)에 cam2 검출이 있으면 → 겹침 → cam3 drop(cam2 채택).
        # cam2가 없으면 → cam3 유니크 영역 → keep. 겹침/유니크를 cam2 존재로 '동적' 판정.
        # 라벨(AV/일반)이 카메라 간 달라도 위치로 dedup → ReID 순간실패에 강건.
        # 단, 매칭된 cam2가 화면 끝에 잘린 검출(지붕만, 하단=프레임 바닥)이면 GPS 무효 →
        # 반대로 cam2를 버리고 cam3(제대로 된 위치)를 채택. (이탈 잔상 중복 방지, 소멸 없음)
        if not bev_reid:
            _FRAME_H = 1080   # 영상 세로 해상도 (1920x1080)
            def _vgrp(c):
                return 0 if c == 0 else (1 if c in (1, 3) else 2)   # 보행자 / 이륜 / 차량
            def _clipped(e):
                return float(e["bbox"][3]) >= _FRAME_H - 10   # bbox 하단이 프레임 바닥에 붙음
            cam2_entries = [(k2, e2) for k2, e2 in enumerate(cctv_traffic) if e2.get("cctv") == 2]
            if cam2_entries:
                _R2 = 64.0  # 8m
                drop_set = set()
                for k, e in enumerate(cctv_traffic):
                    if e.get("cctv") != 3:
                        continue
                    lat, lon = float(e["GPS"][0]), float(e["GPS"][1])
                    g3 = _vgrp(int(e["class"]))
                    match = None
                    for k2, e2 in cam2_entries:
                        if k2 in drop_set or _vgrp(int(e2["class"])) != g3:
                            continue
                        dn = (float(e2["GPS"][0]) - lat) * 111320.0
                        de = (float(e2["GPS"][1]) - lon) * 111320.0 * 0.81
                        if dn * dn + de * de <= _R2:
                            match = k2
                            break
                    if match is None:
                        continue
                    e2 = cctv_traffic[match]
                    if _clipped(e2):
                        # cam2가 잘린 잔상 → cam2 버리고 cam3 유지 (제대로 된 위치)
                        drop_set.add(match)
                        if e2.get("id") == CAV_GLOBAL_ID and e.get("id") != CAV_GLOBAL_ID:
                            cctv_traffic[k]["id"] = CAV_GLOBAL_ID
                    else:
                        # 정상 겹침 → cam3 버리고 cam2 채택
                        drop_set.add(k)
                        if e.get("id") == CAV_GLOBAL_ID and e2.get("id") != CAV_GLOBAL_ID:
                            cctv_traffic[match]["id"] = CAV_GLOBAL_ID
                if drop_set:
                    _before = len(cctv_traffic)
                    cctv_traffic = [e for k, e in enumerate(cctv_traffic) if k not in drop_set]
                    print(f"[OVERLAP] cam2/cam3 겹침 dedup {len(drop_set)}개 drop, {_before}→{len(cctv_traffic)}")
                    ##########정윤성
                    frame_console_logs.append(f"[OVERLAP] cam2/cam3 겹침 dedup {len(drop_set)}개 drop, {_before}→{len(cctv_traffic)}")
                    ##########

        # CAV global_id dedup: 같은 frame에 CAV가 여러 카메라에 동시 출현 시
        # bbox 면적이 가장 큰 카메라의 entry만 남기고 나머지 제거
        # (bev_reid 시엔 BEV가 CAV 포함 전체를 dedup하므로 skip)
        cav_entries = [(idx, e) for idx, e in enumerate(cctv_traffic) if e.get("id") == CAV_GLOBAL_ID]
        if (not bev_reid) and len(cav_entries) > 1:
            def _bbox_area(e):
                b = e["bbox"]
                return float((b[2] - b[0]) * (b[3] - b[1]))
            keep_idx, keep_entry = max(cav_entries, key=lambda x: _bbox_area(x[1]))
            drop_indices = {idx for idx, _ in cav_entries if idx != keep_idx}
            cctv_traffic = [e for i_, e in enumerate(cctv_traffic) if i_ not in drop_indices]
            print(f"[CAVReID] CAV dedup: {len(cav_entries)}개 카메라 동시 출현 → cam {keep_entry['cctv']} 채택 (bbox area 최대)")
            ##########정윤성
            frame_console_logs.append(f"[CAVReID] CAV dedup: {len(cav_entries)}개 카메라 동시 출현 → cam {keep_entry['cctv']} 채택 (bbox area 최대)")
            ##########

        # ── BEV 주행경로 기반 cross-camera ReID + dedup (실험, --bev-reid) ──
        # 활성화 시: 아래 ROI presence-gated dedup 대신 BEV association으로
        # global_id 통합 + 겹침 제거를 한 번에 수행.
        # CAV(88888)도 BEV에 포함시켜 dedup (오인식이 dedup을 빠져나가는 것 방지).
        # 한 그룹에 CAV-tagged 검출이 있으면 대표 검출의 id를 88888로 보존.
        if bev_reid:
            # 주행경로 기반 cross-camera ReID: (cam,id)를 안정 손잡이로 시간누적+락
            link_dets = []
            for _k, e in enumerate(cctv_traffic):
                b = e["bbox"]
                link_dets.append({
                    'key': (int(e["cctv"]), e.get("id")),
                    'lat': float(e["GPS"][0]), 'lon': float(e["GPS"][1]),
                    'cls': int(e["class"]), 'cam': int(e["cctv"]),
                    'bbox_area': abs((b[2] - b[0]) * (b[3] - b[1])),
                    'is_cav': e.get("id") == CAV_GLOBAL_ID,
                    'is_cv': cv_enabled and e.get("id") == CV_GLOBAL_ID, '_k': _k})
            assign, drop_local = bev_tracker.update(link_dets, frame_idx)
            cav_gids = {assign[di] for di, d in enumerate(link_dets)
                        if d['is_cav'] and di in assign}
            cv_gids = {assign[di] for di, d in enumerate(link_dets)
                       if d.get('is_cv') and di in assign}
            for di, d in enumerate(link_dets):
                if di in assign:
                    gid = assign[di]
                    if gid in cav_gids:
                        cctv_traffic[d['_k']]["id"] = CAV_GLOBAL_ID   # CAV 우선(같은 그룹에 둘 다면 CAV)
                    elif gid in cv_gids:
                        cctv_traffic[d['_k']]["id"] = CV_GLOBAL_ID
                    else:
                        cctv_traffic[d['_k']]["id"] = gid
            drop_ks = {link_dets[di]['_k'] for di in drop_local}
            if drop_ks:
                _before = len(cctv_traffic)
                cctv_traffic = [e for k, e in enumerate(cctv_traffic) if k not in drop_ks]
                _msg = (f"[BEV-ReID] 겹침 {len(drop_ks)}개 dedup(락), {_before}->{len(cctv_traffic)} "
                        f"| locked_keys={len(bev_tracker.parent)} active={len(bev_tracker.last_seen)}")
                print(_msg)
                ##########정윤성
                frame_console_logs.append(_msg)
                ##########


        cctv_traffic_dict = {
            "traffic": cctv_traffic,
            "ai_latency": ai_latency,
            "frame_start_time": int(f_start * 1000),
            "frame_end_time": int(time.time() * 1000),
            "frame_idx": frame_idx
        }

        if show_sdsm:
            print_sdsm(cctv_traffic, frame_idx, ai_latency)

        ##########정윤성
        # entry 기능은 현재 버전에서 제거됨 → C++ downstream 구조 보존 위해 빈 배열 유지
        master_tcp_payload = {
            "type": "frame_update",
            "cctv_traffic": cctv_traffic_dict,
            "entry": [],
            "vds": cctv_traffic_vds
        }
        tcp_server.push_result(master_tcp_payload)
        ##########

        _tmp = cctv_traffic_json + ".tmp"
        with open(_tmp, "w", encoding='utf-8') as f:
            json.dump(cctv_traffic_dict, f, indent=4)
        os.replace(_tmp, cctv_traffic_json)
        # Replay log: 매 frame 스냅샷을 한 줄씩 append (JSONL 형식)
        with open(cctv_traffic_log_file, "a", encoding='utf-8') as f:
            f.write(json.dumps({"frame_idx": frame_idx,
                                "traffic": cctv_traffic,
                                "ai_latency": ai_latency}) + "\n")

        _t_json_all = (time.time() - _t_json_start) * 1000
        if not show_sdsm:
            print(f"  [POST] json_write:{_t_json_all:.1f}ms")

        ##########정윤성
        frame_console_logs.append(f"  [POST] json_write:{_t_json_all:.1f}ms")

        logs_payload = {
            "type": "console_logs",
            "logs": frame_console_logs
        }
        tcp_server.push_result(logs_payload)
        ##########

        frame_index += 1

    t = tuple(x / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} '
                f'update per image at shape {{(1, 3, *imgsz)}}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8_hardneg_ped.pt')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x1_0_vehicle.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort')
    parser.add_argument('--source', type=str, default='test.mp4')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.5)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='')
    parser.add_argument('--show-vid', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--save-crop', action='store_true')
    parser.add_argument('--save-vid', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--project', default=WEIGHTS.parent / 'outputs' / 'runs' / 'track')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', default=1, type=int)
    parser.add_argument('--hide-labels', default=False, action='store_true')
    parser.add_argument('--hide-conf', default=False, action='store_true')
    parser.add_argument('--hide-class', default=False, action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--cav-query', type=str, default=str(WEIGHTS.parent / 'datasets' / 'reid_queries' / 'query_image'), help='CAV(목표차량) 쿼리 이미지 폴더')
    parser.add_argument('--cav-threshold', type=float, default=0.8, help='CAV Re-ID 유사도 threshold (max similarity)')
    parser.add_argument('--cav-id', type=int, default=88888, help='CAV로 식별된 객체에 부여할 global_id')
    parser.add_argument('--cv-reid', action='store_true', help='CV(연결차) Re-ID도 함께 실행 (기본 off=CAV만)')
    parser.add_argument('--cv-query', type=str, default=str(WEIGHTS.parent / 'datasets' / 'reid_queries' / 'cv_query_image'), help='CV 쿼리 이미지 폴더')
    parser.add_argument('--cv-threshold', type=float, default=0.78, help='CV Re-ID 유사도 threshold')
    parser.add_argument('--cv-id', type=int, default=77777, help='CV로 식별된 객체에 부여할 global_id')
    parser.add_argument('--reid-clip-weight', type=float, default=None, help='CLIP:색상 히스토그램 비중(0~1). 미지정=기본(fine-tuned 0.8). 흰SUV 구분 어려우면 0.9로(색상↓)')
    parser.add_argument('--no-print-sdsm', dest='show_sdsm', action='store_false', help='SDSM-G 콘솔 출력 끄기 (기본 ON, 끄면 기존 timing 로그 표시)')
    parser.set_defaults(show_sdsm=True)
    parser.add_argument('--frame-skip', type=int, default=1, help='N프레임마다 1번 처리 (1=모두, 4=4프레임당 1개). 실시간 영상속도 맞출 때 사용')
    parser.add_argument('--realtime', action='store_true', help='영상 파일을 RTSP처럼 처리. wall clock 시간보다 뒤처진 frame은 자동 skip (실시간 시뮬레이션용)')
    parser.add_argument('--no-bev-reid', dest='bev_reid', action='store_false', help='BEV 주행경로 기반 dedup 끄기 (기본 ON). 끄면 옛 ROI presence-gated dedup')
    parser.set_defaults(bev_reid=True)
    parser.add_argument('--bev-gate-m', type=float, default=4.0, help='cross-camera 쌍 후보로 볼 GPS 거리(m). 작을수록 빡셈(옆차선 오락↓)')
    parser.add_argument('--bev-overlap-m', type=float, default=5.0, help='cam2↔cam3 겹침 즉시 dedup 거리(m). 같은차 2마커 합침. 크면 옆차선도 합쳐질 위험')
    parser.add_argument('--bev-lock-th', type=float, default=5.0, help='ID쌍 누적점수 임계. 이상이면 동일객체 락. 클수록 느리지만 확실')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
