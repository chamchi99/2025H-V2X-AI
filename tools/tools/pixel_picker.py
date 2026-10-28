#!/usr/bin/env python3
"""
pixel_picker.py — 이미지/영상에서 클릭해 픽셀좌표를 따는 GCP 도구

호스트(디스플레이 있는 환경)에서 실행. Docker 안에선 GUI 안 됨.

사용:
  python3 pixel_picker.py --image frame.jpg
  python3 pixel_picker.py --video datasets/test_scenarios/2026_scenario/scn1_03_cam2_output.mp4 --frame 0
  python3 pixel_picker.py --video ...cam3...mp4 --frame 100 --out cam3_px.txt

조작:
  좌클릭   : 점 추가 (빨간 점 + 번호)
  우클릭   : 마지막 점 취소
  마우스이동: 우상단 돋보기(zoom)로 정밀 위치 확인
  s        : 현재까지 점들을 --out 파일로 저장
  r        : 전체 초기화
  ESC / q  : 종료 (종료 시 점 목록 출력)
"""
import argparse
import cv2
import numpy as np

points = []
cursor = (0, 0)
ZOOM = 6          # 돋보기 배율
ZOOM_W = 220      # 돋보기 창 크기(px)


def render(base):
    img = base.copy()
    # 등록된 점
    for i, (px, py) in enumerate(points):
        cv2.circle(img, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 커서 좌표 + 십자선
    cx, cy = cursor
    cv2.line(img, (cx, 0), (cx, img.shape[0]), (0, 255, 0), 1)
    cv2.line(img, (0, cy), (img.shape[1], cy), (0, 255, 0), 1)
    cv2.putText(img, f"({cx}, {cy})  pts={len(points)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 돋보기 (우상단)
    h, w = base.shape[:2]
    half = ZOOM_W // (2 * ZOOM)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    if x1 > x0 and y1 > y0:
        patch = base[y0:y1, x0:x1]
        zoom = cv2.resize(patch, (ZOOM_W, ZOOM_W), interpolation=cv2.INTER_NEAREST)
        # 돋보기 중앙 십자
        c = ZOOM_W // 2
        cv2.line(zoom, (c, 0), (c, ZOOM_W), (0, 255, 0), 1)
        cv2.line(zoom, (0, c), (ZOOM_W, c), (0, 255, 0), 1)
        cv2.rectangle(zoom, (0, 0), (ZOOM_W - 1, ZOOM_W - 1), (255, 255, 255), 1)
        img[10:10 + ZOOM_W, w - ZOOM_W - 10:w - 10] = zoom
    return img


def main():
    global cursor
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--image', help='이미지 파일')
    g.add_argument('--video', help='영상 파일 (프레임 추출)')
    ap.add_argument('--frame', type=int, default=0, help='--video일 때 프레임 번호')
    ap.add_argument('--out', default='picked_points.txt', help='저장 파일')
    args = ap.parse_args()

    if args.image:
        base = cv2.imread(args.image)
        if base is None:
            print(f"이미지 못 읽음: {args.image}"); return
        title = args.image
    else:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, base = cap.read()
        cap.release()
        if not ok:
            print(f"프레임 읽기 실패: {args.video} #{args.frame}"); return
        title = f"{args.video} #{args.frame}"

    print(f"[픽셀피커] {title}  해상도={base.shape[1]}x{base.shape[0]}")
    print("좌클릭=추가 / 우클릭=취소 / s=저장 / r=초기화 / ESC=종료")

    win = "pixel_picker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, base.shape[1]), min(900, base.shape[0]))

    def cb(event, x, y, flags, _):
        global cursor
        cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            print(f"취소 Point {len(points)}: {points.pop()}")

    cv2.setMouseCallback(win, cb)

    while True:
        cv2.imshow(win, render(base))
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):
            break
        elif k == ord('s'):
            with open(args.out, 'w') as f:
                for i, (px, py) in enumerate(points):
                    f.write(f"{px}, {py}\n")
            print(f"[저장] {len(points)}개 점 → {args.out}")
        elif k == ord('r'):
            points.clear()
            print("[초기화]")

    cv2.destroyAllWindows()
    print("\n=== 수집된 픽셀좌표 ===")
    for i, (px, py) in enumerate(points):
        print(f"  {i+1}: ({px}, {py})")


if __name__ == '__main__':
    main()
