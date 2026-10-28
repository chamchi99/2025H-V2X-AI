"""CAV vs CV 쿼리 분리도 점검 — CLIP+VeRi 유사도로 두 타깃이 구분되는지 정량 확인."""
import importlib.util as ilu, os, cv2, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # codes/
spec = ilu.spec_from_file_location("_reid_module", str(ROOT / "re-id.py"))
mod = ilu.module_from_spec(spec); spec.loader.exec_module(mod)
ext = mod.ReIDExtractor(weights=None, device='cuda')

def load(dirp):
    files = sorted([f for f in Path(dirp).iterdir() if f.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')])
    crops = [cv2.imread(str(f)) for f in files]
    feats = ext.extract_from_crops(crops)
    feats = [feats[i] for i in range(len(feats))]
    return [f.name for f in files], np.array(feats)

QB = str(ROOT.parent / 'datasets' / 'reid_queries')
cav_n, cav_f = load(f'{QB}/query_image')
cv_n,  cv_f  = load(f'{QB}/cv_query_image')

def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def block(A, An, B, Bn):
    return np.array([[cos(a, b) for b in B] for a in A])

cc = block(cav_f, cav_n, cav_f, cav_n)   # CAV-CAV
vv = block(cv_f, cv_n, cv_f, cv_n)       # CV-CV
xv = block(cav_f, cav_n, cv_f, cv_n)     # CAV-CV (교차)

def offdiag_mean(m):
    n = m.shape[0]; s = 0; c = 0
    for i in range(n):
        for j in range(n):
            if i != j: s += m[i][j]; c += 1
    return s/c if c else 0

print(f"CAV 쿼리 {len(cav_n)}장, CV 쿼리 {len(cv_n)}장")
print(f"\n[within] CAV-CAV 평균(자기끼리): {offdiag_mean(cc):.3f}")
print(f"[within] CV-CV  평균(자기끼리): {offdiag_mean(vv):.3f}")
print(f"[cross ] CAV-CV 평균(교차)   : {xv.mean():.3f}  (낮을수록 구분 잘됨)")
print(f"[cross ] CAV-CV 최대(교차)   : {xv.max():.3f}  (threshold 넘으면 오매칭 위험)")
gap = min(offdiag_mean(cc), offdiag_mean(vv)) - xv.mean()
print(f"\n분리 마진(within_min - cross_mean): {gap:.3f}  (+ 클수록 좋음, 음수면 위험)")
print("\n교차 유사도 행렬 (행=CAV, 열=CV):")
print("      " + "  ".join(f"{n[:8]:>8}" for n in cv_n))
for i, an in enumerate(cav_n):
    print(f"{an[:10]:>10} " + "  ".join(f"{xv[i][j]:8.3f}" for j in range(len(cv_n))))
