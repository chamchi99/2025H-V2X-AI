#!/usr/bin/env python3
"""
comm_monitor.py — main.py TCP 송신 검증용 echo 모니터

C++ yolo_receiver와 "동일한 프로토콜"(newline-delimited JSON, :9000)로 접속해서
실제 소켓으로 받은 데이터를 터미널에 그대로 찍어 보여준다.
(C++ 수신기는 frame_update를 화면에 안 띄우므로, 전송 검증/시연용으로 사용)

주의: 서버는 listen(1) 단일 클라이언트라 C++ 수신기와 동시에 못 붙는다.

사용:
  python3 comm_monitor.py --host 172.17.0.2 --port 9000
  python3 comm_monitor.py --host 172.17.0.2 --port 9000 --full   # 전체 JSON raw 출력
  python3 comm_monitor.py --host 172.17.0.2 --port 9000 --max-frames 50
"""
import argparse
import socket
import json

CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'truck'}
CAV_ID = 88888


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='172.17.0.2')
    ap.add_argument('--port', type=int, default=9000)
    ap.add_argument('--max-frames', type=int, default=0, help='0=무한, N=N프레임 받고 종료')
    ap.add_argument('--full', action='store_true', help='수신 raw JSON 전체 출력')
    ap.add_argument('--logs', action='store_true', help='console_logs도 출력')
    args = ap.parse_args()

    import time as _t
    buf = b''
    n_frame = 0
    s = None
    try:
        while True:
            # (재)연결 — C++ 수신기처럼 끊기면 재시도 (startup 큐-empty 블립 대응)
            if s is None:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((args.host, args.port))
                    print(f"[MONITOR] {args.host}:{args.port} 연결 성공! 수신 데이터 출력 시작\n" + "=" * 70)
                except OSError:
                    print(f"\r[MONITOR] {args.host}:{args.port} 연결 대기...", end='', flush=True)
                    _t.sleep(1.0)
                    s = None
                    continue
            chunk = s.recv(65536)
            if not chunk:
                print("\n[MONITOR] 연결 끊김 → 재연결 시도")
                s.close(); s = None; buf = b''
                continue
            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    continue
                mtype = msg.get('type')

                if mtype == 'frame_update':
                    n_frame += 1
                    ct = msg.get('cctv_traffic', {})
                    traffic = ct.get('traffic', [])
                    fidx = ct.get('frame_idx')
                    lat = ct.get('ai_latency')
                    vds = msg.get('vds', [])

                    if args.full:
                        print(json.dumps(msg, ensure_ascii=False, indent=2))
                        continue

                    print(f"\n┌─ frame_update  #{fidx}  | 객체 {len(traffic)}개 "
                          f"| ai_latency {lat}s | vds {len(vds)}건")
                    for o in traffic:
                        oid = o.get('id')
                        cls = CLASS_NAMES.get(int(o.get('class', -1)), '?')
                        tag = 'CAV' if oid == CAV_ID else cls
                        g = o.get('GPS', [None, None])
                        sp = o.get('speed')
                        sp_s = '-' if sp in (None, 999) else f'{abs(sp):.0f}km/h'
                        hd = o.get('heading')
                        hd_s = '-' if hd in (None, 999) else f'{hd:.2f}rad'
                        print(f"│  cam{o.get('cctv')} {tag:<8} id={str(oid):<8} "
                              f"GPS=({g[0]:.6f},{g[1]:.6f}) spd={sp_s:<7} hd={hd_s} "
                              f"stop={o.get('stop_duration')}s")
                    if vds:
                        for v in vds:
                            print(f"│  [VDS] cam{v.get('cctv')} "
                                  f"l1={v.get('lane1_speed')}km/h({v.get('lane1_count')}) "
                                  f"l2={v.get('lane2_speed')}km/h({v.get('lane2_count')}) "
                                  f"l3={v.get('lane3_speed')}km/h({v.get('lane3_count')}) "
                                  f"merge={v.get('merge_count')}")
                    print("└" + "─" * 60)

                elif mtype == 'console_logs' and args.logs:
                    for lg in msg.get('logs', []):
                        print(f"  [LOG] {lg}")

                if args.max_frames and n_frame >= args.max_frames:
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        if s is not None:
            s.close()
    print(f"\n[MONITOR] 종료. 총 frame_update {n_frame}개 수신.")


if __name__ == '__main__':
    main()
