# 다중 CCTV 기반 실시간 교통 객체 인지 및 BEV 통합 표출 시스템

ETRI 원내 도로에 설치된 다중 CCTV 영상으로부터 교통 객체를 실시간 검출·추적하고,
검출 결과를 GPS 좌표로 변환하여 단일 Bird's Eye View(BEV) 글로벌 맵에 통합 표출하는
V2X 인프라 측 인지 시스템이다.

---

## 연구개발 개요

이기종 V2X(Vehicle-to-Everything) 환경에서 도로 인프라(CCTV)가 자율주행 차량의 인지 범위를
보조·확장하도록, 교차로 및 사각지대의 교통 상황을 실시간으로 검출·추적하여 통합 모니터링
정보로 제공하는 것을 목표로 한다. 단일 카메라가 아닌 다중 CCTV를 융합하여 넓은 영역을
일관된 좌표계 위에서 다루며, 인프라가 인지한 교통 객체 정보를 자율주행 차량 및 관제 측에
전달할 수 있는 형태로 가공한다.

- 다중 CCTV 기반 실시간 교통 객체(차량·보행자·자전거) 검출 및 다중 객체 추적
- 카메라 픽셀 좌표를 실세계 GPS 좌표로 변환하는 인프라 캘리브레이션 및 좌표 융합
- 겹치는 카메라 화각에서의 동일 객체 중복 제거(다중 카메라 융합)
- 특정 자율주행 차량(CAV)의 영상 기반 재식별 및 연속 추적
- 검출 결과의 BEV 글로벌 맵 실시간 통합 표출 및 외부 연동

---

## ⚙️ 개발 환경 설정

개발과 추론은 모두 Docker 컨테이너(`KETI_AI`) 환경에서 수행한다. 검출·추적 모델은
Ultralytics(YOLOv8)와 PyTorch 기반으로 구성되며, NVIDIA GPU 환경에서 동작한다.
컨테이너 내부에 파이프라인 코드와 가중치(`weights/`), 카메라 캘리브레이션 설정이 함께
배치되어 있어, 컨테이너를 띄운 뒤 메인 파이프라인을 실행하면 RTSP 스트림 또는 영상 파일을
입력으로 곧바로 동작한다. 세부 의존성은 `requirements.txt` / `environment.yml`에 정리되어 있다.

> 처음 환경을 구성한다면 아래 순서로 진행한다.
> 1. NVIDIA 드라이버 + Docker(+ NVIDIA Container Toolkit)가 설치된 GPU 서버를 준비한다.
> 2. 파이프라인 코드와 가중치가 들어 있는 작업 디렉터리를 컨테이너의 `/workspace`로 마운트한다.
> 3. 아래 [실행 방법](#️-실행-방법)대로 메인 스크립트를 실행하고 BEV 맵을 연다.

---

## ▶️ 실행 방법

### 1) 입력 준비
파이프라인은 **3대 카메라(cam1·cam2·cam3)를 동시에** 처리한다. 실행 시에는 **cam2(교차로 전경)
입력 하나만** 지정하면, 나머지 cam1·cam3는 경로(파일명)의 카메라 토큰을 자동 치환하여 찾는다.
즉 파일명에 `cam2` / `camera2` / `ch02` / `cctv02` 중 하나가 들어 있어야 하며, 같은 위치에
대응되는 cam1·cam3 파일이 있어야 한다.

실시간 CCTV 입력의 경우에도 cam2에 해당하는 스트림의 RTSP 주소(xxx.xxx.xxx.82)를 입력하면, 
나머지 cam1(xxx.xxx.xxx.81)과 cam3(xxx.xxx.xxx.83)에 해당하는 RTSP 스트림을 자동으로 불러온다. 

```
예) 입력으로  scn1_01_cam2_output.mp4  를 주면
    자동으로  scn1_01_cam1_output.mp4 (cam1),  scn1_01_cam3_output.mp4 (cam3) 를 함께 로드
```

- **실시간**: RTSP 스트림 주소
- **영상 파일**: ETRI 시나리오 영상(`2026_scenario/` 등)

### 2) 파이프라인 실행
```bash
# 영상 파일 입력 (cam2 경로만 지정 → cam1/cam3 자동 매핑)
docker exec -w /workspace KETI_AI \
  python traffic_tracker.py \
    --source 2026_scenario/scn1_01_cam2_output.mp4 \
    --yolo-weights weights/yolov8_hardneg_ped.pt \
    --save-vid

# 실시간 RTSP 입력
docker exec -w /workspace KETI_AI \
  python traffic_tracker.py --source "rtsp://129.254.222.82" --realtime

# 종료
docker exec KETI_AI pkill -f traffic_tracker.py
```

**자주 쓰는 옵션**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source` | 입력(RTSP 주소 또는 cam2 영상 경로) | — |
| `--yolo-weights` | 검출 가중치 | `weights/yolov8_hardneg_ped.pt` |
| `--conf-thres` | 검출 신뢰도 임계값 | `0.5` |
| `--cav-query` | 목표 차량(CAV) 재식별용 쿼리 이미지 폴더 | `query_image` |
| `--cav-threshold` | CAV 재식별 유사도 임계값 | `0.75` |
| `--frame-skip` | N프레임마다 1번 처리(실시간 속도 보정용) | `1` |
| `--realtime` | 영상 파일을 RTSP처럼 wall-clock에 맞춰 처리 | off |
| `--save-vid` | 검출·추적이 그려진 결과 영상 저장 | off |
| `--show-vid` | 처리 화면 실시간 표시 | off |

### 3) 결과(BEV 맵) 확인
실행하면 작업 디렉터리(`/workspace`)에 인지 결과 `cctv-traffic.json`이 매 프레임 갱신된다.
BEV 글로벌 맵(`etri_mapbox.html`)은 같은 디렉터리의 이 파일을 읽어 표출하므로, 간단한 로컬
웹 서버로 열면 된다.

```bash
# 작업 디렉터리에서
python3 -m http.server 8000
# 브라우저에서 → http://localhost:8000/etri_mapbox.html
```

---

## 🔄 처리 파이프라인

한 프레임이 입력에서 BEV 맵 표출까지 거치는 과정은 다음과 같다.

| 단계 | 처리 내용 | 핵심 산출 |
|------|-----------|-----------|
| ① 입력 | cam1·cam2·cam3 3채널 프레임을 동기 로드 | 원본 프레임 |
| ② 검출 | YOLOv8로 프레임별 5종 객체 검출 | 클래스 + bbox |
| ③ 추적 | 카메라별 StrongSORT/OCSort로 추적 ID 부여 | track ID |
| ④ CAV 재식별 | 쿼리 이미지와 CLIP+VeRi 특징 비교로 목표 차량 식별 | CAV global_id |
| ⑤ 좌표 변환 | bbox 기준점을 카메라별 Homography로 GPS 변환 | 위경도(GPS) |
| ⑥ 상태 추정 | GPS 시계열로 속도·진행 방향 산출, 정지 차량 고정, 보행자 영역 판정 | speed/heading 등 |
| ⑦ 다중 카메라 융합 | 동일 객체 중복 제거(CAV 통합 + cam2/cam3 겹침 ROI 정리) | 중복 제거된 객체 목록 |
| ⑧ 출력 | 프레임 인지 결과를 JSON으로 기록 | `cctv-traffic.json`, `…-log.jsonl` |
| ⑨ 표출/연동 | BEV 맵이 JSON을 읽어 실시간 표출(필요 시 소켓 송신) | 글로벌 맵 화면 |

```
[cam1·cam2·cam3]
      │  ② YOLOv8 검출
      ▼
  ③ 추적(StrongSORT/OCSort) ──④ CAV 재식별(CLIP+VeRi)
      │
      ▼  ⑤ Homography → GPS,  ⑥ 속도/방향/정지/보행자영역
  ⑦ 다중 카메라 융합(중복 제거)
      │
      ▼  ⑧ cctv-traffic.json
  ⑨ BEV 글로벌 맵 표출 (+ 외부 연동)
```

---

## 📊 데이터 수집, 파싱, 학습

**검출 환경** : ETRI 원내 도로에 설치된 CCTV 3대(도로 진입부, 교차로 전경, 교차로
확대)의 1920×1080 영상을 프레임 단위로 파싱하여 사용한다.

**검출 클래스** : Person, Cyclist, Car, Motorcycle, Truck (5종)

**데이터 가공·학습**
- 사전 학습 모델로 프레임을 1차 자동 라벨링한 뒤 검수·병합하여 학습 데이터셋(GT)을 구성
- 학습/검증 데이터를 분리하고 클래스 체계를 5종으로 정렬하여 데이터셋 구성
- 자전거·보행자 등 약점 클래스를 보강하기 위한 보조 데이터셋(하드 네거티브 포함)을 별도 운용
- COCO 사전학습 `YOLOv8m`을 ETRI 원내 도로 데이터셋(2025·2026)으로 fine-tuning하여
  현장 환경에 맞춘 검출 모델을 확보

또한 ETRI 원내 도로에서 차량·자전거·보행자 중심의 다양한 교통 상황을 재현한 시험 시나리오
영상을 카메라별로 취득하여, 검출·추적 및 좌표 융합 성능 검증에 활용한다.

---

## 🗺️ 교통 모니터링 서비스

CCTV 3채널 영상을 입력으로 받아, 검출 → 추적 → 좌표 변환 → 융합 → 표출의 흐름으로
동작한다.

**객체 검출 및 추적**
- YOLOv8 기반으로 5종 교통 객체를 실시간 검출
- StrongSORT/OCSort로 객체별 추적 ID를 부여하여 프레임 간 동일성 유지
- GPS 시계열을 이용해 객체의 속도·진행 방향을 추정하고, 정지 차량은 표시 위치를 고정

**검출 객체 위치를 글로벌 맵에 표출**
- 각 CCTV 영상의 픽셀 좌표 → 카메라별 Homography → 실세계 GPS 좌표
- 3대 카메라의 검출 결과를 단일 위성 기반 BEV 글로벌 맵 위에 통합 표출
- 차선(Lane) geometry를 함께 오버레이하여 교통 상황을 직관적으로 확인

<table>
  <tr>
    <td align="center" valign="bottom">
      <img
        width="868"
        height="724"
        alt="위성사진"
        src="https://github.com/user-attachments/assets/e91e1e29-a365-4500-b114-154e6eff7c70"
      />
    </td>
    <td align="center" valign="bottom">
      <img
        width="868"
        height="724"
        alt="도로사진"
        src="https://github.com/user-attachments/assets/1eac8b04-72ff-4f4b-aa99-ca6f3c4dde16"
      />
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <sub>글로벌 맵 표출 예시 (좌: 위성사진, 우: 도로사진)</sub>
    </td>
  </tr>
</table>

**다중 카메라 융합 (겹침영역 중복 제거)**
- 교차로 전경 카메라와 확대(zoom) 카메라의 화각이 겹쳐, 동일 객체가 두 카메라에서
  별개로 표출되는 문제가 발생
- 확대 카메라의 실제 커버영역(ROI)을 캘리브레이션으로 산출하고, 그 영역에 들어오는
  광역 카메라 검출을 확대 카메라 기준으로 정리하여 중복을 제거

**특정 차량(CAV) 재식별 기반 추적**
- 목표 자율주행 차량의 쿼리 이미지와 검출 객체를 CLIP+VeRi 특징으로 비교하여 식별
- 여러 카메라에 동시에 잡힌 동일 CAV는 하나로 통합하여 표출

<img width="1825" height="1037" alt="image" src="https://github.com/user-attachments/assets/73d72e0c-65b4-45e7-abfa-fa60b2b1d057" />





---

## 🛰️ 시스템 출력 및 연동

매 프레임의 인지 결과는 객체의 위치(GPS)·클래스·추적 ID·속도·방향 등을 담은 정형 데이터로
생성되며, BEV 글로벌 맵이 이를 주기적으로 받아 실시간으로 표출한다. 동일 결과는 외부
시스템(관제·V2X) 연동을 위해 소켓으로도 송신할 수 있다.

- 실시간 인지 결과 : `cctv-traffic.json` (BEV 맵 표출 및 외부 송신)
- 리플레이 로그 : `cctv-traffic-log.jsonl` (프레임별 스냅샷 누적)

<table>
  <tr>
    <td align="center" valign="middle" height="550">
      <img
        width="641"
        alt="CCTV 분석 결과"
        src="https://github.com/user-attachments/assets/e1b60092-4de1-4378-81da-791deeecbaec"
      />
    </td>
    <td align="center" valign="middle" height="550">
      <img
        width="199"
        alt="교통 정보 UI"
        src="https://github.com/user-attachments/assets/271e374a-262a-42bd-a8dd-4d72448766ee"
      />
    </td>
  </tr>
  <tr>
    <td align="center">
      <sub>CCTV 영상 내 차량 이동정보 분석 결과물 (cctv-traffic.json)</sub>
    </td>
    <td align="center">
      <sub>교통 정보 표출 예시 (UI/UX 화면)</sub>
    </td>
  </tr>
</table>
---

## 결과물

**주요 산출물**
- 다중 CCTV 기반 실시간 교통 객체 검출·추적
- 카메라 영상 기반 검출 객체의 GPS 위치 추정 및 BEV 글로벌 맵 통합 표출
- 다중 카메라 겹침영역 중복 제거(카메라 간 좌표 융합)
- 트랜스포머 기반 특정 차량(CAV) 재식별 및 연속 추적
- ETRI 원내 도로 환경에 fine-tuning된 YOLOv8 검출 모델

**성능·검증**
- YOLOv8 기반 검출 모델의 클래스별 검출 정확도(mAP) 및 실시간 처리 성능 검증
- ETRI 원내 도로 시험 시나리오 영상을 통한 검출·추적·좌표 융합 결과 확인
- 상세 결과 및 분석은 `docs/`(논문 초안, 지연시간/속도·방향 분석 등) 참조


<img width="1401" height="601" alt="image" src="https://github.com/user-attachments/assets/c636a632-2ed9-430f-98f2-eafd8631fd5d" />




