"""cam202(toWGS) / cam203(toWGS_ch3) ROI를 homography로 GPS에 투영하고
겹침영역(overlap polygon)을 계산한다. (shapely 없이 numpy만 사용)

- 이미지 1920x1080 그리드를 GPS로 투영
- REF에서 너무 먼(투영 발산) 점은 제외
- 각 카메라 coverage를 convex hull로 근사 → Sutherland-Hodgman으로 교집합
- 결과를 GeoJSON으로 저장해 글로벌맵에서 눈으로 확인 가능
"""
import numpy as np
import json
import math

REF_LAT, REF_LON = 36.38234, 127.3667
W, H_IMG = 1920, 1080
MAX_DIST_M = 250.0  # REF에서 이 거리 넘는 투영점은 발산으로 보고 제외
COS_LAT = math.cos(math.radians(REF_LAT))


def toWGS(x1, y1):  # CAM202 (cam2)
    H = [[ 1.00984829e-01,  2.55985472e-01, -2.14652241e+01],
         [-3.21637351e-02, -2.40964641e-01,  2.22981797e+02],
         [-4.64864032e-04,  2.00501356e-02,  1.00000000e+00]]
    R1 = np.matmul(H, [x1, y1, 1]); R1 /= R1[2]
    return (REF_LAT + R1[1] / 111320,
            REF_LON + R1[0] / (111320 * COS_LAT))


def toWGS_ch3(x1, y1):  # CAM203 (cam3)
    H = [[ 5.79818214e-03,  8.74080988e-03,  9.19854631e+00],
         [-7.23231478e-04, -1.20114616e-02,  1.26411286e+00],
         [ 4.46590396e-06,  7.35272439e-04,  1.00000000e+00]]
    R1 = np.matmul(H, [x1, y1, 1]); R1 /= R1[2]
    return (REF_LAT + R1[1] / 111320,
            REF_LON + R1[0] / (111320 * COS_LAT))


def latlon_to_m(lat, lon):
    """REF 기준 local meter (east, north)."""
    e = (lon - REF_LON) * 111320.0 * COS_LAT
    n = (lat - REF_LAT) * 111320.0
    return e, n


def m_to_latlon(e, n):
    lat = REF_LAT + n / 111320.0
    lon = REF_LON + e / (111320.0 * COS_LAT)
    return lat, lon


def project_grid(fn, step=20):
    """이미지 그리드를 meter로 투영, 발산점 제외."""
    pts = []
    for px in range(0, W + 1, step):
        for py in range(0, H_IMG + 1, step):
            lat, lon = fn(px, py)
            e, n = latlon_to_m(lat, lon)
            if math.hypot(e, n) <= MAX_DIST_M:
                pts.append((e, n))
    return np.array(pts)


def convex_hull(points):
    """Andrew's monotone chain. points: Nx2 → hull vertices CCW."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def poly_area(poly):
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        a += x1*y2 - x2*y1
    return abs(a) / 2.0


def clip_polygon(subject, clip):
    """Sutherland-Hodgman. clip은 볼록(CCW) 가정."""
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0
    def intersect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = a; x4, y4 = b
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))
    output = list(subject)
    cn = len(clip)
    for i in range(cn):
        a = clip[i]; b = clip[(i+1) % cn]
        inp = output; output = []
        if not inp:
            break
        s = inp[-1]
        for p in inp:
            if inside(p, a, b):
                if not inside(s, a, b):
                    output.append(intersect(s, p, a, b))
                output.append(p)
            elif inside(s, a, b):
                output.append(intersect(s, p, a, b))
            s = p
    return output


def hull_to_geojson_ring(hull):
    ring = []
    for e, n in hull:
        lat, lon = m_to_latlon(e, n)
        ring.append([lon, lat])  # GeoJSON = [lon, lat]
    if ring:
        ring.append(ring[0])
    return ring


g2 = project_grid(toWGS)
g3 = project_grid(toWGS_ch3)
h2 = convex_hull(g2)
h3 = convex_hull(g3)
overlap = clip_polygon(h2, h3)

a2, a3 = poly_area(h2), poly_area(h3)
ao = poly_area(overlap) if len(overlap) >= 3 else 0.0

print(f"cam202 coverage hull: {len(h2)} verts, area={a2:.0f} m^2")
print(f"cam203 coverage hull: {len(h3)} verts, area={a3:.0f} m^2")
print(f"overlap polygon:      {len(overlap)} verts, area={ao:.0f} m^2")
if a2 and a3:
    print(f"overlap / cam202 = {ao/a2*100:.1f}%   overlap / cam203 = {ao/a3*100:.1f}%")

# overlap 중심/범위
if len(overlap) >= 3:
    ov = np.array(overlap)
    cen_e, cen_n = ov[:,0].mean(), ov[:,1].mean()
    clat, clon = m_to_latlon(cen_e, cen_n)
    print(f"overlap centroid: lat={clat:.6f}, lon={clon:.6f}")
    print(f"overlap extent: east [{ov[:,0].min():.0f},{ov[:,0].max():.0f}]m  north [{ov[:,1].min():.0f},{ov[:,1].max():.0f}]m")

geo = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"cam": 202, "fill": "#ff0000"},
     "geometry": {"type": "Polygon", "coordinates": [hull_to_geojson_ring(h2)]}},
    {"type": "Feature", "properties": {"cam": 203, "fill": "#0000ff"},
     "geometry": {"type": "Polygon", "coordinates": [hull_to_geojson_ring(h3)]}},
]}
if len(overlap) >= 3:
    geo["features"].append(
        {"type": "Feature", "properties": {"cam": "overlap", "fill": "#00ff00"},
         "geometry": {"type": "Polygon", "coordinates": [hull_to_geojson_ring(overlap)]}})

with open("cam_roi_overlap.geojson", "w") as f:
    json.dump(geo, f, indent=2)
print("→ wrote _tools/cam_roi_overlap.geojson")
