# maps/ — 지도 / 지리 데이터

- `etri_mapbox.html` — **실사용 지도** (Mapbox satellite). 객체를 실시간 표출
- `etri_vworld.html` — VWorld 배경 지도 버전
- `etri_lanes.geojson` — 차선 geometry

## 실행 (⚠️ 서버는 HV2X 루트에서 띄울 것)
지도 HTML은 데이터를 **루트기준 절대경로**로 fetch한다:
- `/outputs/cctv-traffic.json`, `/outputs/cctv-traffic-log.jsonl` (main.py 실행 산출물)
- `/maps/etri_lanes.geojson`

따라서 **HTTP 서버를 반드시 HV2X 루트에서** 실행해야 경로가 맞는다:
```bash
cd /home/super/HV2X          # ← 루트에서!
python3 -m http.server 8888
# 브라우저: http://<호스트>:8888/maps/etri_mapbox.html
```
(main.py가 outputs/에 json을 계속 갱신 → 지도가 폴링해서 실시간 반영)
