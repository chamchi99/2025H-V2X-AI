import ffmpeg
import os

# ── 경로 설정 ──────────────────────────────────────────────
input_dir  = "/workspace/datasets/test_scenarios/2026_scenario/260709_laneoccupancy/raw_videos/"
output_dir = "/workspace/datasets/test_scenarios/2026_scenario/260709_laneoccupancy/cropped_videos/"

# ── 캠별 자를 구간: (파일명, 시작초 ss, 끝초 to) ──────────────
#   세 캠 싱크가 안 맞으므로 캠마다 시작/끝을 따로 지정한다.
clips = [
    ("1-cam1.avi", 99,181),
    ("1-cam2.avi", 96,178),
    ("1-cam3.avi", 94,176),
]

# ── 실행 ───────────────────────────────────────────────────
os.makedirs(output_dir, exist_ok=True)
for fname, ss, to in clips:
    in_path  = os.path.join(input_dir, fname)
    stem     = os.path.splitext(fname)[0]                # 확장자 제외 파일명
    out_path = f"{output_dir}{stem}_output2.mp4"
    print(f"[CUT] {fname}  {ss}s ~ {to}s  ->  {out_path}", flush=True)
    # map='0:v:0' → h264 영상 스트림만 선택 (깨진 rawvideo 스트림 무시)
    ffmpeg.input(in_path).output(out_path, ss=ss, to=to, map='0:v:0').run(overwrite_output=True)
    print(f"  done: {out_path}", flush=True)

print("ALL DONE")