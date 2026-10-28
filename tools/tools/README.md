# tools/ — 유틸리티 (실행 파이프라인에 직접 안 물림)

- `gcp_check.py` — GCP 오차 점검(RMS·Leave-One-Out·H코드). `python gcp_check.py --px cam*_px.txt --gps cam*_gps.txt --cam camN`
- `pixel_picker.py` — 영상에서 GCP 픽셀좌표 클릭 취득(돋보기)
- `calib_homography.py` — GCP 클릭→Homography 계산(GCPS 하드코딩)
- 화면 구역(검출 마스크·차선·보행자 구역) 정의는 `lane_define/define_zones.html`로 통합 (아래 하위 폴더 참고)
- `videocut.py` / `hstack.py` — 동영상 자르기 / 가로결합
- `comm_monitor.py` — TCP(:9000) 수신 모니터(C++ 수신기 대체 검증)
- `SendSocket.py` — 소켓 송신 유틸
- `train_reid.py` — Re-ID 학습
- `roi_overlap.py` + `roi_overlap_view.html` + `cam_roi_overlap.geojson` — 카메라 ROI 겹침 계산/시각화

### 하위 폴더
- `build_engine/` — TensorRT 엔진 빌드: `build_engine.py`, `build_engine_b3.py`, `build_engine_dynamic.py`, `build_trt_b3.py`
- `training/` — 검출/ReID 학습·평가: `train_gt.py`, `eval_gt.py`, `prepare_gt_dataset.py`, `reid_bench.py`
- `lane_define/` — 구역/BEV 정의 HTML 툴: `define_zones.html`(cam1/cam2/cam3 통합 — 보행자 구역 + cam1 차선 점유율), `bev_map_v4.1.html`
- `dev_scripts/` — 개발/벤치 스크립트: `track_yolo26.py`, `track_dev_comb_benchmark_v10_rotated_demo_vid.py`
