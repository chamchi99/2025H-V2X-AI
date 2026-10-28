"""
호모그래피 캘리브레이션 툴
- GCP GPS 좌표는 고정, 영상에서 픽셀 위치만 클릭
- 출력: main.py의 toWGS / toWGS_ch1 / toWGS_ch3 에 붙여넣을 코드

사용법:
  python calib_homography.py --video <영상파일> --cam <cam1|cam2|cam3> [--frame <프레임번호>]

조작:
  좌클릭   : 현재 안내 중인 GCP 픽셀 위치 등록
  우클릭   : 마지막 점 취소
  Enter    : H행렬 계산 및 결과 출력
  ESC      : 종료
"""

import cv2
import numpy as np
import argparse

REF_LAT = 36.38234
REF_LON = 127.3667

# ── GCP 정의 (GPS 고정) ──────────────────────────────────────────────
GCPS = {
    'cam1': [
        (36.38230120, 127.36668908),  # GCP-1
        (36.38235821, 127.36668811),  # GCP-2
        (36.38230115, 127.36640534),  # GCP-3
        (36.38236032, 127.36640555),  # GCP-4
        (36.38230303, 127.36624639),  # GCP-5
        (36.38236162, 127.36624581),  # GCP-6
    ],
    'cam2': [  # 2026-06-23 재측정
        (36.38233063, 127.36685364),  # GCP-1
        (36.38233149, 127.36688069),  # GCP-2
        (36.38235470, 127.36686200),  # GCP-3
        (36.38235470, 127.36689290),  # GCP-4
        (36.38238440, 127.36691330),  # GCP-5
        (36.38238440, 127.36702980),  # GCP-6
        (36.38245800, 127.36695100),  # GCP-7
        (36.38245800, 127.36697380),  # GCP-8
    ],
    'cam3': [  # 2026-06-23 재측정
        (36.38228014, 127.36684021),  # GCP-1
        (36.38228014, 127.36686192),  # GCP-2
        (36.38229400, 127.36684280),  # GCP-3
        (36.38229400, 127.36686674),  # GCP-4
        (36.38233063, 127.36685364),  # GCP-5
        (36.38233149, 127.36688069),  # GCP-6
        (36.38231800, 127.36689320),  # GCP-7
        (36.38230300, 127.36689840),  # GCP-8
    ],
}

CAM_FUNC = {'cam1': 'toWGS_ch1', 'cam2': 'toWGS', 'cam3': 'toWGS_ch3'}

def gps_to_meters(lat, lon):
    x = (lon - REF_LON) * 111320 * np.cos(np.radians(REF_LAT))
    y = (lat - REF_LAT) * 111320
    return x, y

pts_pixel = []
base_frame = None
cam = None

def current_gcp_idx():
    return len(pts_pixel)

def draw(frame):
    img = frame.copy()
    gcps = GCPS[cam]
    total = len(gcps)
    done = len(pts_pixel)

    # 완료된 점
    for i, (px, py) in enumerate(pts_pixel):
        cv2.circle(img, (px, py), 8, (0, 255, 0), -1)
        cv2.putText(img, f"GCP-{i+1}", (px+10, py-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # 안내 텍스트
    if done < total:
        lat, lon = gcps[done]
        guide = f"GCP-{done+1} 클릭  ({lat:.8f}, {lon:.8f})"
        cv2.putText(img, guide, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(img, f"진행: {done}/{total}  |  우클릭=취소  |  ESC=종료",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        cv2.putText(img, f"모든 GCP 완료 ({total}/{total})  |  Enter=계산  |  ESC=종료",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

    return img

def mouse_cb(event, x, y, flags, _):
    gcps = GCPS[cam]
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = current_gcp_idx()
        if idx >= len(gcps):
            print("모든 GCP가 이미 등록됐습니다. Enter로 계산하세요.")
            return
        pts_pixel.append((x, y))
        lat, lon = gcps[idx]
        print(f"GCP-{idx+1} ({lat:.8f}, {lon:.8f}) → 픽셀 ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if pts_pixel:
            removed = pts_pixel.pop()
            print(f"GCP-{len(pts_pixel)+1} 취소: 픽셀 {removed}")

def compute_and_print():
    gcps = GCPS[cam]
    n = len(pts_pixel)
    if n < 4:
        print(f"최소 4개 필요 (현재 {n}개)")
        return

    src = np.float64(pts_pixel[:n])
    dst = np.float64([gps_to_meters(lat, lon) for lat, lon in gcps[:n]])

    H, status = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None:
        print("H행렬 계산 실패. 점 분포를 확인하세요.")
        return

    func_name = CAM_FUNC[cam]
    print("\n" + "="*65)
    print(f"  main.py → {func_name}() 의 H 교체")
    print("="*65)
    print(f"    H = [[{H[0,0]:.8e}, {H[0,1]:.8e}, {H[0,2]:.8e}],")
    print(f"         [{H[1,0]:.8e}, {H[1,1]:.8e}, {H[1,2]:.8e}],")
    print(f"         [{H[2,0]:.8e}, {H[2,1]:.8e}, {H[2,2]:.8e}]]")
    print("="*65)

    print("\n  [검증] 기준점별 역변환 오차")
    for i, ((px, py), (lat, lon)) in enumerate(zip(pts_pixel[:n], gcps[:n])):
        R1 = np.matmul(H, [px, py, 1])
        R1 /= R1[2]
        calc_lat = REF_LAT + R1[1] / 111320
        calc_lon = REF_LON + R1[0] / (111320 * np.cos(np.radians(REF_LAT)))
        dy = (calc_lat - lat) * 111320
        dx = (calc_lon - lon) * 111320 * np.cos(np.radians(REF_LAT))
        err = np.sqrt(dx**2 + dy**2)
        print(f"  GCP-{i+1}: 오차 {err:.3f} m")

def main():
    global base_frame, cam
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--cam',   required=True, choices=['cam1','cam2','cam3'])
    parser.add_argument('--frame', type=int, default=100)
    args = parser.parse_args()
    cam = args.cam

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, base_frame = cap.read()
    cap.release()
    if not ret:
        print(f"프레임 {args.frame} 읽기 실패")
        return

    print(f"카메라: {cam}  |  GCP {len(GCPS[cam])}개  |  프레임: {args.frame}")
    print("화면에 안내되는 GCP 순서대로 해당 위치를 클릭하세요.\n")

    cv2.namedWindow('calib', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('calib', 1280, 720)
    cv2.setMouseCallback('calib', mouse_cb)

    while True:
        cv2.imshow('calib', draw(base_frame))
        key = cv2.waitKey(20) & 0xFF
        if key == 13:   # Enter
            compute_and_print()
        elif key == 27: # ESC
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
