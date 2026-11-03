"""
GCP 좌표 추출 스크립트
사용법: python extract_gcp.py <ubx파일경로> [ubx파일경로2] ...
예시:  python extract_gcp.py gcp1.ubx gcp2.ubx gcp3.ubx
"""

import sys
import statistics as stats
from pyubx2 import UBXReader

def extract_gcp(path, hacc_threshold=20):
    rows = []
    with open(path, 'rb') as f:
        ubr = UBXReader(f)
        for raw, parsed in ubr:
            if parsed is None or parsed.identity != 'NAV-PVT':
                continue
            try:
                if parsed.carrSoln == 2 and parsed.hAcc <= hacc_threshold:
                    rows.append({
                        'lat': parsed.lat,
                        'lon': parsed.lon,
                        'hMSL': parsed.hMSL / 1000.0,
                        'hAcc': parsed.hAcc,
                    })
            except AttributeError:
                continue

    if not rows:
        return None

    return {
        'n': len(rows),
        'lat': stats.mean(r['lat'] for r in rows),
        'lon': stats.mean(r['lon'] for r in rows),
        'alt': stats.mean(r['hMSL'] for r in rows),
        'hAcc_mean': stats.mean(r['hAcc'] for r in rows),
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python extract_gcp.py gcp1.ubx gcp2.ubx ...")
        sys.exit(1)

    files = sys.argv[1:]
    print(f"{'파일명':30s} {'n':>5s} {'위도':>14s} {'경도':>14s} {'고도(m)':>9s} {'hAcc평균':>9s}")
    print("-" * 90)

    for i, path in enumerate(files, 1):
        result = extract_gcp(path)
        filename = path.split('/')[-1].split('\\')[-1]
        if result is None:
            print(f"{filename:30s}  Fixed 데이터 없음 (hAcc>{20}mm 이거나 NAV-PVT 미기록)")
        else:
            print(f"{filename:30s} {result['n']:5d} {result['lat']:.9f} {result['lon']:.9f} {result['alt']:9.3f}m {result['hAcc_mean']:8.1f}mm")
