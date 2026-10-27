"""
Vehicle Re-Identification System
다중 CCTV 기반 차량 재인식 기술개발 산출물

사용법:
  # 쿼리 이미지로 특정 차량 탐색 (단일 카메라)
  python re-id.py --query query/ --source cam1.mp4 --yolo-weights best.pt --reid-weights osnet.pt

  # 크로스 카메라 탐색 (cam1 → cam2 순차)
  python re-id.py --query query/ --source cam1.mp4 --source2 cam2.mp4 --yolo-weights best.pt --reid-weights osnet.pt

  # 쿼리 없이 갤러리 자동 구축 (기존 방식)
  python re-id.py --source cam1.mp4 --yolo-weights best.pt --reid-weights osnet.pt --open-gallery
"""

import argparse
import numpy as np
import cv2
import time
import csv
import os
import torch
from pathlib import Path
from ultralytics import YOLO
import sys
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# train_reid(VehicleReIDModel 정의)는 재편으로 codes/tools 로 이동됨 → import 위해 경로 추가
_tools_dir = str(Path(ROOT) / 'tools')
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)
if str(Path(ROOT) / 'yolov5') not in sys.path:
    sys.path.insert(0, str(Path(ROOT) / 'yolov5'))

# ============================================================
# 1. ReID 특징 추출기
# ============================================================
class ReIDExtractor:
    def __init__(self, weights=None, device='cuda'):
        self.device = device
        self.use_clip = False
        self.use_osnet = False
        self.clip_weight = None   # None=기본(0.8/0.7). CLI로 덮어쓰면 그 값 사용(CLIP:색상 비중)

        # CLIP 모델 로드 시도 (VeRi fine-tuned 우선)
        try:
            import open_clip
            model_name = 'ViT-B-32'
            pretrained = 'laion2b_s34b_b79k'
            clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )

            # Fine-tuned weight 로드
            # weights는 top-level(codes의 부모)로 이동됨 → ROOT/weights 아니라 부모/weights
            ft_path = os.path.join(os.path.dirname(ROOT), 'weights', 'reid', 'clip_veri_best.pt')
            if os.path.exists(ft_path):
                from train_reid import VehicleReIDModel
                self.clip_model = VehicleReIDModel(clip_model, embed_dim=512, num_classes=576)
                ckpt = torch.load(ft_path, map_location=device, weights_only=False)
                self.clip_model.clip_visual.load_state_dict(ckpt['clip_visual'])
                self.clip_model.proj.load_state_dict(ckpt['proj'])
                self.clip_model.eval()
                self.clip_model.to(device)
                self._clip_finetuned = True
                print(f"[ReID] CLIP+VeRi fine-tuned 로드 완료 (mAP={ckpt.get('mAP',0):.3f})")
            else:
                self.clip_model = clip_model
                self.clip_model.eval()
                self._clip_finetuned = False
                print(f"[ReID] CLIP pretrained 로드 (fine-tuned weight 없음)")

            self.clip_model = self.clip_model.to(device)
            self.use_clip = True
        except Exception as e:
            print(f"[ReID] CLIP 로드 실패: {e}")
            # fallback: OSNet
            if weights and os.path.exists(weights):
                try:
                    from trackers.strong_sort.reid_multibackend import ReIDDetectMultiBackend
                    self.model = ReIDDetectMultiBackend(Path(weights), device=device)
                    self.use_osnet = True
                    print(f"[ReID] OSNet fallback 로드 완료: {weights}")
                except Exception as e2:
                    print(f"[ReID] OSNet도 실패, 히스토그램만 사용: {e2}")
            else:
                print("[ReID] 히스토그램 기반 특징 추출만 사용")

    def _clip_preprocess_crops(self, crops):
        """CLIP 전처리: BGR crop → CLIP 텐서"""
        from PIL import Image
        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensors.append(self.clip_preprocess(pil_img))
        return torch.stack(tensors).to(self.device)

    def extract_from_crops(self, crops):
        """crop 이미지 리스트에서 특징 벡터 추출"""
        if len(crops) == 0:
            return np.array([])

        # 색상 히스토그램 특징
        color_feats = self._color_features(crops)
        color_norms = np.linalg.norm(color_feats, axis=1, keepdims=True)
        color_norms[color_norms == 0] = 1
        color_feats = color_feats / color_norms

        if self.use_clip:
            with torch.no_grad():
                batch = self._clip_preprocess_crops(crops)
                if self._clip_finetuned:
                    clip_feats, _ = self.clip_model(batch)  # already L2-normed
                    clip_feats = clip_feats.cpu().numpy().astype(np.float32)
                else:
                    clip_feats = self.clip_model.encode_image(batch).cpu().numpy().astype(np.float32)
            if not self._clip_finetuned:
                clip_norms = np.linalg.norm(clip_feats, axis=1, keepdims=True)
                clip_norms[clip_norms == 0] = 1
                clip_feats = clip_feats / clip_norms
            # CLIP + 색상 가중 결합. clip_weight 지정 시 그 값, 아니면 기본(fine-tuned 0.8 / pretrained 0.7)
            cw = self.clip_weight if self.clip_weight is not None else (0.8 if self._clip_finetuned else 0.7)
            features = np.concatenate([clip_feats * cw, color_feats * (1 - cw)], axis=1)
        elif self.use_osnet:
            osnet_feats = self.model(crops).detach().cpu().numpy()
            osnet_norms = np.linalg.norm(osnet_feats, axis=1, keepdims=True)
            osnet_norms[osnet_norms == 0] = 1
            osnet_feats = osnet_feats / osnet_norms
            features = np.concatenate([osnet_feats * 0.6, color_feats * 0.4], axis=1)
        else:
            features = color_feats

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return features / norms

    def extract(self, frame, bboxes):
        """프레임 + bbox 목록에서 특징 벡터 추출"""
        crops = []
        valid_idx = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            crops.append(crop)
            valid_idx.append(i)

        if len(crops) == 0:
            return np.array([]), []

        features = self.extract_from_crops(crops)
        return features, valid_idx

    def _color_features(self, crops):
        """색상 히스토그램 특징 (HSV 상/하 분할 + 명도)"""
        features = []
        for crop in crops:
            resized = cv2.resize(crop, (64, 128))
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            h = resized.shape[0] // 2
            upper = hsv[:h]
            lower = hsv[h:]
            hist_upper = cv2.calcHist([upper], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
            hist_lower = cv2.calcHist([lower], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
            feat = np.concatenate([hist_upper, hist_lower, v_hist])
            features.append(feat)
        return np.array(features)


# ============================================================
# 2. Gallery 관리
# ============================================================
class Gallery:
    def __init__(self):
        self.entries = {}  # {id: {feature, name, class, count}}
        self.next_id = 1

    def register(self, feature, name=None, cls=None, query_features=None):
        """차량 등록. name이 있으면 쿼리 차량."""
        gid = self.next_id
        self.next_id += 1
        self.entries[gid] = {
            'feature': feature.copy(),
            'name': name or f'vehicle_{gid}',
            'class': cls,
            'is_query': name is not None,
            'count': 1,
            'found_frames': [],
            'query_features': query_features,  # 개별 피처 리스트 (max 매칭용), None이면 mean 단일 벡터 사용
        }
        return gid

    def query(self, feature, threshold=0.6, cls_filter=None):
        """갤러리에서 가장 유사한 차량 검색"""
        best_id = None
        best_sim = -1

        for gid, entry in self.entries.items():
            if cls_filter is not None and entry['class'] is not None and entry['class'] != cls_filter:
                continue
            sim = float(np.dot(feature, entry['feature']))
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        if best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim

    def update(self, gid, feature, alpha=0.9):
        """EMA 방식 특징 벡터 갱신 (쿼리는 갱신하지 않음)"""
        if self.entries[gid]['is_query']:
            self.entries[gid]['count'] += 1
            return
        old = self.entries[gid]['feature']
        new = alpha * old + (1 - alpha) * feature
        norm = np.linalg.norm(new)
        if norm > 0:
            new = new / norm
        self.entries[gid]['feature'] = new
        self.entries[gid]['count'] += 1

    def size(self):
        return len(self.entries)

    def query_entries(self):
        """쿼리로 등록된 항목만 반환"""
        return {gid: e for gid, e in self.entries.items() if e['is_query']}


# ============================================================
# 3. Re-ID 엔진
# ============================================================
class VehicleReIDEngine:
    VEHICLE_CLASSES = {1, 2, 3, 4}  # bicycle, car, motorcycle, truck
    CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'truck'}
    COLORS = {
        0: (52, 211, 153), 1: (6, 182, 212), 2: (239, 68, 68),
        3: (234, 179, 8), 4: (180, 100, 60)
    }
    FOUND_COLOR = (0, 255, 0)      # 쿼리 매칭 성공
    NOT_FOUND_COLOR = (128, 128, 128)  # 미매칭

    def __init__(self, yolo_weights, reid_weights=None, device='cuda',
                 conf_thres=0.5, similarity_threshold=0.6, open_gallery=False):

        print(f"[Engine] YOLO 로드: {yolo_weights}")
        self.yolo = YOLO(str(yolo_weights))
        self.yolo.to(device)
        self.conf_thres = conf_thres
        self.sim_threshold = similarity_threshold
        self.open_gallery = open_gallery

        self.extractor = ReIDExtractor(reid_weights, device)
        self.gallery = Gallery()

        self.log = []
        self.frame_idx = 0
        self.current_source = ''

    def load_queries(self, query_dir):
        """쿼리 등록.
        - query_dir/ 안에 하위 폴더가 있으면 폴더마다 별도 차량으로 등록
          예) query/black_bmw/*.jpg, query/white_ferrari/*.jpg → 2개 쿼리
        - 하위 폴더 없이 이미지만 있으면 전체를 단일 쿼리로 등록
        각 차량의 이미지들은 피처 평균을 내서 하나의 쿼리 엔트리로 등록.
        """
        query_dir = Path(query_dir)
        exts = ('.jpg', '.jpeg', '.png', '.bmp')

        subdirs = sorted([d for d in query_dir.iterdir() if d.is_dir()])
        if subdirs:
            targets = [(d.name, sorted([f for f in d.iterdir() if f.suffix.lower() in exts]))
                       for d in subdirs]
        else:
            images = sorted([f for f in query_dir.iterdir() if f.suffix.lower() in exts])
            targets = [(query_dir.name, images)]

        registered = 0
        for name, images in targets:
            if not images:
                print(f"[Query] '{name}' 폴더에 이미지 없음, 건너뜀")
                continue
            all_features = []
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                features = self.extractor.extract_from_crops([img])
                if len(features) > 0:
                    all_features.append(features[0])
            if not all_features:
                print(f"[Query] '{name}' 피처 추출 실패")
                continue
            mean_feature = np.mean(all_features, axis=0)
            mean_feature = mean_feature / (np.linalg.norm(mean_feature) + 1e-6)
            gid = self.gallery.register(mean_feature, name=name, query_features=all_features)
            mode = f"max-of-{len(all_features)}" if len(all_features) > 1 else "단일"
            print(f"[Query] '{name}' — {len(all_features)}장 {mode} 매칭으로 등록 (ID={gid})")
            registered += 1

        print(f"[Query] 총 {registered}개 쿼리 등록 완료")

    def process_frame(self, frame):
        """단일 프레임 처리"""
        self.frame_idx += 1

        results = self.yolo.predict(frame, conf=self.conf_thres, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        dets = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        vehicle_mask = np.array([c in self.VEHICLE_CLASSES for c in clses])
        if not vehicle_mask.any():
            return []

        v_dets = dets[vehicle_mask]
        v_confs = confs[vehicle_mask]
        v_clses = clses[vehicle_mask]

        features, valid_idx = self.extractor.extract(frame, v_dets)
        if len(features) == 0:
            return []

        # 쿼리당 top-1 매칭: 각 쿼리에 대해 가장 유사한 검출 1개만 FOUND
        query_entries = self.gallery.query_entries()
        matched_det_indices = set()  # 이미 매칭된 검출 인덱스
        top1_matches = {}  # {query_gid: (det_i, similarity)}

        if query_entries:
            for qid, qentry in query_entries.items():
                best_i, best_sim = None, -1
                all_sims = []
                for i, idx in enumerate(valid_idx):
                    qfeats = qentry.get('query_features')
                    if qfeats and len(qfeats) > 1:
                        sim = float(max(np.dot(features[i], qf) for qf in qfeats))
                    else:
                        sim = float(np.dot(features[i], qentry['feature']))
                    all_sims.append((i, idx, sim))
                    if i not in matched_det_indices and sim > best_sim and sim >= self.sim_threshold:
                        best_sim = sim
                        best_i = i
                # 디버그: 전체 유사도 출력 (상위 5개)
                all_sims.sort(key=lambda x: x[2], reverse=True)
                if self.frame_idx <= 5 or self.frame_idx % 100 == 0:
                    top5 = ', '.join([f'det{s[0]}={s[2]:.3f}' for s in all_sims[:5]])
                    chosen = f'→ best=det{best_i}({best_sim:.3f})' if best_i is not None else '→ no match'
                    print(f"  [DBG #{self.frame_idx}] query={qentry['name']}: {top5} {chosen}")
                if best_i is not None:
                    top1_matches[qid] = (best_i, best_sim)
                    matched_det_indices.add(best_i)

        output = []
        for i, idx in enumerate(valid_idx):
            feat = features[i]
            cls = int(v_clses[idx])
            bbox = v_dets[idx]
            conf = float(v_confs[idx])

            # top-1 매칭 결과 확인
            matched_qid = None
            sim = 0.0
            for qid, (mi, ms) in top1_matches.items():
                if mi == i:
                    matched_qid = qid
                    sim = ms
                    break

            if matched_qid is not None:
                entry = self.gallery.entries[matched_qid]
                self.gallery.update(matched_qid, feat)
                action = 'FOUND'
                reid_id = matched_qid
                reid_name = entry['name']
                entry['found_frames'].append({
                    'frame': self.frame_idx,
                    'source': self.current_source,
                    'bbox': bbox.tolist(),
                    'similarity': float(sim),
                })
            elif self.open_gallery:
                reid_id = self.gallery.register(feat, cls=cls)
                action = 'NEW'
                reid_name = f'vehicle_{reid_id}'
                sim = 0.0
            else:
                action = 'NONE'
                reid_id = -1
                reid_name = ''
                # 가장 높은 쿼리 유사도 표시 (디버깅용)
                _, sim = self.gallery.query(feat, threshold=0.0)
                sim = float(sim)

            result = {
                'id': reid_id,
                'name': reid_name,
                'cls': cls,
                'cls_name': self.CLASS_NAMES.get(cls, 'unknown'),
                'bbox': bbox.tolist(),
                'conf': conf,
                'similarity': float(sim),
                'action': action,
            }
            output.append(result)

            self.log.append({
                'frame': self.frame_idx,
                'source': self.current_source,
                'id': reid_id,
                'name': reid_name,
                'class': self.CLASS_NAMES.get(cls, 'unknown'),
                'x1': int(bbox[0]), 'y1': int(bbox[1]),
                'x2': int(bbox[2]), 'y2': int(bbox[3]),
                'conf': round(conf, 3),
                'similarity': round(float(sim), 3),
                'action': action,
            })

        return output

    def draw_results(self, frame, results):
        """결과를 프레임에 시각화"""
        for r in results:
            x1, y1, x2, y2 = map(int, r['bbox'])

            if r['action'] == 'FOUND':
                color = self.FOUND_COLOR
                label = f"FOUND:{r['name']} sim={r['similarity']:.2f}"
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), color, 4)
            elif r['action'] == 'NEW':
                color = self.COLORS.get(r['cls'], (255, 255, 255))
                label = f"ID:{r['id']} {r['cls_name']}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                color = self.NOT_FOUND_COLOR
                label = f"{r['cls_name']} sim={r['similarity']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            text_color = (0, 0, 0) if r['action'] == 'FOUND' else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # 상단 상태
        queries = self.gallery.query_entries()
        found_names = [e['name'] for e in queries.values() if len(e['found_frames']) > 0]
        status = f"Source: {self.current_source} | Frame: {self.frame_idx} | Query: {len(queries)} | Found: {len(found_names)}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def save_log(self, path):
        if not self.log:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.log[0].keys())
            writer.writeheader()
            writer.writerows(self.log)
        print(f"[Engine] 로그 저장: {path} ({len(self.log)}건)")

    def print_summary(self):
        total = len(self.log)
        found = sum(1 for l in self.log if l['action'] == 'FOUND')
        queries = self.gallery.query_entries()

        print(f"\n{'='*50}")
        print(f"[결과 요약]")
        print(f"  총 검출: {total}건")
        print(f"  쿼리 매칭 (FOUND): {found}건")
        print(f"\n[쿼리별 결과]")
        for gid, entry in queries.items():
            frames = entry['found_frames']
            if frames:
                sources = set(f['source'] for f in frames)
                avg_sim = np.mean([f['similarity'] for f in frames])
                print(f"  {entry['name']}: {len(frames)}회 검출, 평균 유사도 {avg_sim:.3f}, 카메라: {sources}")
            else:
                print(f"  {entry['name']}: 미검출")


# ============================================================
# 4. 소스 처리
# ============================================================
def process_source(engine, source, save_video=False, show=False, output_dir='runs/reid'):
    """단일 소스(영상/이미지폴더/단일이미지) 처리"""
    source = str(source)
    source_name = Path(source).stem
    engine.current_source = source_name

    is_video = source.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
    is_dir = os.path.isdir(source)
    is_image = source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

    if is_video:
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_video:
            out_path = os.path.join(output_dir, f'reid_{source_name}.mp4')
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"[Main] 영상 저장: {out_path}")

        print(f"[Main] 영상: {source} ({total_frames}프레임, {fps}fps)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = engine.process_frame(frame)
            frame = engine.draw_results(frame, results)
            if writer:
                writer.write(frame)
            if show:
                cv2.imshow('Re-ID', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if engine.frame_idx % 100 == 0:
                found = sum(1 for l in engine.log if l['action'] == 'FOUND')
                print(f"  프레임 {engine.frame_idx}/{total_frames}, FOUND: {found}건")

        cap.release()
        if writer:
            writer.release()

    elif is_dir:
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        images = sorted([f for f in Path(source).iterdir() if f.suffix.lower() in exts])
        print(f"[Main] 이미지 폴더: {source} ({len(images)}장)")

        out_img_dir = os.path.join(output_dir, f'images_{source_name}')
        os.makedirs(out_img_dir, exist_ok=True)

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            results = engine.process_frame(frame)
            frame = engine.draw_results(frame, results)
            cv2.imwrite(os.path.join(out_img_dir, img_path.name), frame)
            if show:
                cv2.imshow('Re-ID', frame)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        print(f"[Main] 결과 이미지 저장: {out_img_dir}")

    elif is_image:
        print(f"[Main] 이미지: {source}")
        frame = cv2.imread(source)
        if frame is not None:
            results = engine.process_frame(frame)
            frame = engine.draw_results(frame, results)
            out_path = os.path.join(output_dir, f'reid_{os.path.basename(source)}')
            cv2.imwrite(out_path, frame)
            print(f"[Main] 결과 저장: {out_path}")


# ============================================================
# 5. 메인 실행
# ============================================================
def run(source, source2=None, query=None, yolo_weights='best.pt',
        reid_weights=None, device='cuda', conf_thres=0.5,
        sim_threshold=0.6, open_gallery=False,
        save_video=False, show=False, output_dir='runs/reid'):

    os.makedirs(output_dir, exist_ok=True)

    engine = VehicleReIDEngine(
        yolo_weights=yolo_weights,
        reid_weights=reid_weights,
        device=device,
        conf_thres=conf_thres,
        similarity_threshold=sim_threshold,
        open_gallery=open_gallery,
    )

    # 쿼리 등록
    if query:
        engine.load_queries(query)
    elif not open_gallery:
        print("[Warning] --query 미지정, --open-gallery도 꺼져있음. 매칭 대상이 없습니다.")
        print("  --query <폴더> 또는 --open-gallery 옵션을 사용하세요.")
        return

    # source1 처리
    print(f"\n{'='*50}")
    print(f"[Source 1] {source}")
    print(f"{'='*50}")
    process_source(engine, source, save_video, show, output_dir)

    # source2 처리 (cross-camera)
    if source2:
        print(f"\n{'='*50}")
        print(f"[Source 2 - Cross Camera] {source2}")
        print(f"  갤러리 유지: {engine.gallery.size()}개 항목")
        print(f"{'='*50}")
        process_source(engine, source2, save_video, show, output_dir)

    if show:
        cv2.destroyAllWindows()

    engine.save_log(os.path.join(output_dir, 'reid_log.csv'))
    engine.print_summary()


def parse_args():
    parser = argparse.ArgumentParser(description='Vehicle Re-ID System')
    parser.add_argument('--source', type=str, required=True,
                        help='영상/이미지폴더/이미지 경로 (camera 1)')
    parser.add_argument('--source2', type=str, default=None,
                        help='두 번째 소스 경로 (camera 2, cross-camera)')
    parser.add_argument('--query', type=str, default=None,
                        help='쿼리 이미지 폴더 (탐색 대상 차량)')
    parser.add_argument('--yolo-weights', type=str, default='best.pt')
    parser.add_argument('--reid-weights', type=str, default=None,
                        help='OSNet 가중치 경로')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--conf-thres', type=float, default=0.5)
    parser.add_argument('--sim-threshold', type=float, default=0.6,
                        help='유사도 임계값 (0~1)')
    parser.add_argument('--open-gallery', action='store_true',
                        help='미매칭 차량도 갤러리에 자동 등록')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output-dir', type=str, default='runs/reid')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))
