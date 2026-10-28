"""영상 → 이미지 프레임 추출 (반자동 레이블링용 데이터 파싱).

일정 간격마다 프레임을 뽑아 이미지로 저장한다. avi/mp4 등 OpenCV가 읽는
모든 컨테이너를 입력으로 받는다.

예)
  python extract_frames.py --source clip.avi --interval 30 --out ./raw_images
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser(description="영상에서 일정 간격으로 프레임을 추출한다.")
    ap.add_argument("--source", required=True, help="입력 영상 경로 (avi/mp4 등)")
    ap.add_argument("--out", required=True, help="프레임 이미지를 저장할 폴더")
    ap.add_argument("--interval", type=int, default=30,
                    help="몇 프레임마다 저장할지 (기본 30)")
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"],
                    help="저장 이미지 확장자 (기본 jpg)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.source}")

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.source))[0]

    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.interval == 0:
            name = f"{stem}_frame_{idx:06d}.{args.ext}"
            cv2.imwrite(os.path.join(args.out, name), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"완료: {idx} 프레임 중 {saved} 장 저장 → {args.out}")


if __name__ == "__main__":
    main()
