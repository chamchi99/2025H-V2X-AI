import os, glob

# 전제: ML Backend를 프로젝트 모델(yolov8_hardneg_ped.pt)로 띄우고,
#       XML 라벨을 person / cyclist / car / motorcycle / truck 로 설정한 경우.
#
# Label Studio는 XML 순서와 무관하게 클래스 번호를 "알파벳순"으로 매긴다.
#   → 내보내기 번호: car=0, cyclist=1, motorcycle=2, person=3, truck=4
#
# 이 프로젝트 학습 클래스 순서: 0:person 1:cyclist 2:car 3:motorcycle 4:truck
# 아래는 (내보내기 번호 → 프로젝트 번호) 매핑.
mapping = {
    0: 2,   # car        → 2
    1: 1,   # cyclist    → 1
    2: 3,   # motorcycle → 3
    3: 0,   # person     → 0
    4: 4,   # truck      → 4
}

#label_dir = "/home/super/HV2X/datasets/train_datasets/새폴더/labels"
label_dir = "/home/super/HV2X/datasets/train_datasets/mixed_cyclist_reproduced_laneoccadded/labels"
for txt in glob.glob(os.path.join(label_dir, "*.txt")):
    lines = open(txt).readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        old_cls = int(parts[0])
        parts[0] = str(mapping[old_cls])
        new_lines.append(" ".join(parts) + "\n")
    open(txt, "w").writelines(new_lines)

print("변환완료")
