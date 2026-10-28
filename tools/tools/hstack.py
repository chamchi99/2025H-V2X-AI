import ffmpeg
import sys

if len(sys.argv) != 4:
    print("사용법: python hstack.py <왼쪽영상> <오른쪽영상> <출력파일>")
    sys.exit(1)

left  = sys.argv[1]
right = sys.argv[2]
out   = sys.argv[3]

l = ffmpeg.input(left)
r = ffmpeg.input(right)
ffmpeg.filter([l, r], 'hstack').output(out).run()
print(f"저장 완료: {out}")
