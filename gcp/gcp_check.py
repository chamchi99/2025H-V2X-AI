"""
GCP 오차 즉시 점검 스크립트 (현장용)
- 픽셀좌표 파일 + GPS좌표 파일 → Homography 계산 → 점별 오차 / RMS / 최대오차
- Leave-One-Out 교차검증: 각 점을 빼고 나머지로 H를 만들어 그 점을 예측 →
  "측정이 의심스러운 점"을 즉시 가려냄 (현장에서 재측정 판단)
- main.py에 붙일 H 코드도 출력 (또는 --apply 로 자동 교체)

사용법:
  python gcp_check.py --px cam2_px.txt --gps cam2_gps.txt --cam cam2
  python gcp_check.py --px cam3_px.txt --gps cam3_gps.txt --cam cam3
  # 특정 픽셀의 위경도 확인(현장 임의점 검증):
  python gcp_check.py --px cam2_px.txt --gps cam2_gps.txt --cam cam2 --test-px 960 540
  # RMS 목표(0.15m) 통과 시 main.py의 toWGS* H 자동 교체(백업 .bak 생성):
  python gcp_check.py --px cam2_px.txt --gps cam2_gps.txt --cam cam2 --apply
  #   목표 초과여도 강제 적용: --apply --force

파일 형식(유연 — 숫자만 추출):
  px : 한 줄에  x, y          (예:  488, 949)
  gps: 한 줄에  lat, lon      (예:  36.3823, 127.3668  또는  {lat:36.3823, lon:127.3668},)
  주석/기호 섞여도 됨. lat은 36~37, lon은 127~128 범위로 자동 인식.
"""

import argparse
import re
import shutil
import datetime
import numpy as np
import cv2

REF_LAT = 36.38234
REF_LON = 127.36670

# 카메라별 traffic_tracker 함수명 (출력용)
CAM_FUNC = {'cam1': 'toWGS_ch1', 'cam2': 'toWGS', 'cam3': 'toWGS_ch3'}

# 권장 임계 (실험계획서 목표치)
TH_RMS = 0.15        # 전체 RMS 목표 (m)
TH_POINT = 1.0       # 점별 오차 경고 (m)
TH_LOO = 2.0         # Leave-One-Out 예측오차 경고 — 넘으면 재측정 권장 (m)


def gps_to_meters(lat, lon):
    x = (lon - REF_LON) * 111320 * np.cos(np.radians(REF_LAT))
    y = (lat - REF_LAT) * 111320
    return x, y


def meters_to_gps(x, y):
    lat = REF_LAT + y / 111320
    lon = REF_LON + x / (111320 * np.cos(np.radians(REF_LAT)))
    return lat, lon


def _floats(line):
    """주석/기호 무시하고 부동소수 숫자만 추출."""
    return [float(t) for t in re.findall(r'-?\d+\.?\d*', line)]


def load_px(path):
    pts = []
    for ln in open(path, encoding='utf-8'):
        f = _floats(ln)
        if len(f) >= 2:
            pts.append((f[0], f[1]))
    return pts


def load_gps(path):
    """lat(36~37), lon(127~128) 범위로 자동 인식 → GCP 번호 같은 잡숫자 무시."""
    pts = []
    for ln in open(path, encoding='utf-8'):
        f = _floats(ln)
        lat = next((v for v in f if 36.0 <= v <= 37.0), None)
        lon = next((v for v in f if 127.0 <= v <= 128.0), None)
        if lat is not None and lon is not None:
            pts.append((lat, lon))
    return pts


def compute_H(px, gps):
    src = np.float64(px)
    dst = np.float64([gps_to_meters(la, lo) for la, lo in gps])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    return H


def reproj_error(H, px, gps):
    """픽셀 → H → 위경도 → 실측 위경도 거리(m)."""
    R = H @ np.array([px[0], px[1], 1.0])
    R /= R[2]
    la, lo = meters_to_gps(R[0], R[1])
    dy = (la - gps[0]) * 111320
    dx = (lo - gps[1]) * 111320 * np.cos(np.radians(REF_LAT))
    return np.hypot(dx, dy)


def apply_to_main(main_py, fn, H, cam, n, rms):
    """main.py의 fn() 함수 안 H 블록 + 캘리브 주석줄만 교체. 백업 후 in-place 수정."""
    src = open(main_py, encoding='utf-8').read()

    m_def = re.search(rf'^def {re.escape(fn)}\(', src, re.MULTILINE)
    if not m_def:
        print(f"  ✗ {main_py} 에 def {fn}( 없음 — 교체 취소"); return False
    # 대상 함수 범위: def fn( ~ 다음 최상위 def 직전
    m_next = re.search(r'^def ', src[m_def.end():], re.MULTILINE)
    seg_s = m_def.start()
    seg_e = m_def.end() + m_next.start() if m_next else len(src)
    seg = src[seg_s:seg_e]

    # H 블록 (들여쓰기 보존) 매칭
    m_h = re.search(r'(?m)^([ \t]*)H = \[\[.*?\]\]', seg, re.DOTALL)
    if not m_h:
        print(f"  ✗ {fn}() 안에 H = [[ ... ]] 블록 없음 — 교체 취소"); return False
    ind = m_h.group(1)
    today = datetime.date.today().isoformat()
    new_h = (f"{ind}H = [[ {H[0,0]:.8e}, {H[0,1]:.8e}, {H[0,2]:.8e}],\n"
             f"{ind}     [{H[1,0]:.8e}, {H[1,1]:.8e}, {H[1,2]:.8e}],\n"
             f"{ind}     [{H[2,0]:.8e}, {H[2,1]:.8e}, {H[2,2]:.8e}]]")
    new_seg = seg[:m_h.start()] + new_h + seg[m_h.end():]

    # def 바로 다음 주석줄(캘리브 이력) 갱신 — 있으면 교체, 없으면 삽입
    note = f"    # {cam} — {today} gcp_check 자동적용 ({n} GCP, RMS {rms:.3f}m)"
    m_cmt = re.match(r'(def [^\n]*\n)([ \t]*#[^\n]*\n)', new_seg)
    if m_cmt:
        new_seg = new_seg[:m_cmt.start(2)] + note + "\n" + new_seg[m_cmt.end(2):]
    else:
        m_hd = re.match(r'(def [^\n]*\n)', new_seg)
        new_seg = new_seg[:m_hd.end()] + note + "\n" + new_seg[m_hd.end():]

    shutil.copy2(main_py, main_py + '.bak')
    open(main_py, 'w', encoding='utf-8').write(src[:seg_s] + new_seg + src[seg_e:])
    print(f"  ✓ {fn}() H 교체 완료  (백업: {main_py}.bak)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--px', required=True)
    ap.add_argument('--gps', required=True)
    ap.add_argument('--cam', default='cam2', choices=['cam1', 'cam2', 'cam3'])
    ap.add_argument('--test-px', nargs=2, type=float, metavar=('X', 'Y'),
                    help='이 픽셀의 위경도를 출력(현장 임의점 검증)')
    ap.add_argument('--apply', action='store_true',
                    help='RMS 목표 통과 시 main.py의 toWGS* 함수 H를 자동 교체')
    ap.add_argument('--main-py', default='/workspace/codes/main.py',
                    help='교체 대상 main.py 경로 (기본: /workspace/codes/main.py)')
    ap.add_argument('--force', action='store_true',
                    help='RMS 목표 초과여도 --apply 강제 실행')
    args = ap.parse_args()

    px = load_px(args.px)
    gps = load_gps(args.gps)
    n = min(len(px), len(gps))
    if len(px) != len(gps):
        print(f"  ⚠ 픽셀 {len(px)}개 / GPS {len(gps)}개 — 개수 불일치, 앞 {n}개만 사용")
    px, gps = px[:n], gps[:n]
    if n < 4:
        print(f"  ✗ 최소 4점 필요 (현재 {n})"); return

    print(f"\n{'='*60}\n  GCP 점검: {args.cam}  ({n}점)\n{'='*60}")

    # 1) 전체 H + 점별 재투영 오차
    H = compute_H(px, gps)
    errs = [reproj_error(H, px[i], gps[i]) for i in range(n)]
    rms = float(np.sqrt(np.mean(np.square(errs))))
    print("\n[1] 재투영 오차 (전체 점으로 H 계산)")
    for i, e in enumerate(errs):
        flag = '  ⚠' if e > TH_POINT else ''
        print(f"    GCP-{i+1:<2}: {e:.3f} m{flag}")
    print(f"    ── RMS {rms:.3f} m / 최대 {max(errs):.3f} m  "
          f"(목표 RMS<{TH_RMS})  {'✓' if rms < TH_RMS else '✗ 초과'}")

    # 2) Leave-One-Out: 각 점 빼고 나머지로 H → 그 점 예측 (측정 신뢰성)
    print("\n[2] Leave-One-Out 교차검증 (그 점 빼고 예측 → 측정 의심점 색출)")
    loo = []
    for i in range(n):
        oth = [j for j in range(n) if j != i]
        if len(oth) < 4:
            loo.append(None); continue
        Hi = compute_H([px[j] for j in oth], [gps[j] for j in oth])
        e = reproj_error(Hi, px[i], gps[i])
        loo.append(e)
        flag = '  ⚠ 재측정 권장' if e > TH_LOO else ''
        print(f"    GCP-{i+1:<2}: {e:.3f} m{flag}")
    valid = [e for e in loo if e is not None]
    if valid:
        worst = int(np.argmax([e if e is not None else -1 for e in loo]))
        print(f"    ── LOO 평균 {np.mean(valid):.3f} m / 최악 GCP-{worst+1} {loo[worst]:.3f} m")

    # 3) traffic_tracker용 H 코드
    fn = CAM_FUNC[args.cam]
    print(f"\n[3] main.py → {fn}() 의 H 교체")
    print(f"    H = [[{H[0,0]:.8e}, {H[0,1]:.8e}, {H[0,2]:.8e}],")
    print(f"         [{H[1,0]:.8e}, {H[1,1]:.8e}, {H[1,2]:.8e}],")
    print(f"         [{H[2,0]:.8e}, {H[2,1]:.8e}, {H[2,2]:.8e}]]")

    # 4) 임의 픽셀 검증
    if args.test_px:
        R = H @ np.array([args.test_px[0], args.test_px[1], 1.0]); R /= R[2]
        la, lo = meters_to_gps(R[0], R[1])
        print(f"\n[4] 픽셀 ({args.test_px[0]:.0f},{args.test_px[1]:.0f}) → "
              f"위경도 ({la:.8f}, {lo:.8f})")

    # 5) main.py 자동 교체 (--apply)
    if args.apply:
        print(f"\n[5] main.py 자동 교체 → {fn}()")
        if rms >= TH_RMS and not args.force:
            print(f"    ✗ RMS {rms:.3f}m ≥ 목표 {TH_RMS}m — 교체 취소 "
                  f"(강제하려면 --force)")
        else:
            if rms >= TH_RMS:
                print(f"    ⚠ RMS {rms:.3f}m 목표 초과지만 --force 로 강제 적용")
            apply_to_main(args.main_py, fn, H, args.cam, n, rms)
    print()


if __name__ == '__main__':
    main()
