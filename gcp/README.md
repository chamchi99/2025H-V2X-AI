# gcp/ — 지상기준점(GCP) 측정 데이터

- `camN_px.txt` — GCP 픽셀좌표 (한 줄 `x, y`)
- `camN_gps.txt` — GCP 실측 위경도 (한 줄 `lat, lon`; 순서가 px와 일치해야 함)
- `_gcp_frames/` — GCP 선정용 프레임 캡처

점검: `python ../codes/tools/gcp_check.py --px camN_px.txt --gps camN_gps.txt --cam camN`

