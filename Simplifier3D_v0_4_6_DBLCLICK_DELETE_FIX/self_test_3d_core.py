from pathlib import Path
from geometry3d_core import GeometryPoint3D, _distance_point_segment_xyz, load_xyz_trajectory, simplify_xyz
ROOT = Path(__file__).resolve().parent

def main():
    d = _distance_point_segment_xyz(0,1,0, 0,0,0, 0,0,2)
    assert abs(d-1.0) < 1e-12, d
    pts = load_xyz_trajectory(ROOT/'model_data'/'line3d.csv')
    assert len(simplify_xyz(pts, 0.0)) == 2
    pts = load_xyz_trajectory(ROOT/'model_data'/'polyline3d.csv')
    assert len(simplify_xyz(pts, 0.0)) == 4
    pts = [
        GeometryPoint3D(0,0,0,0), GeometryPoint3D(1,1,0,0),
        GeometryPoint3D(2,2,0,0,'M1','keep'), GeometryPoint3D(3,3,0,0),
        GeometryPoint3D(4,4,0,0),
    ]
    out = simplify_xyz(pts, 1000.0)
    assert len(out) == 3 and out[1].marker == 'M1'
    print('geometry3d_core self-test: OK')
    print('line3d -> 2 points')
    print('polyline3d -> 4 points')
    print('marker preservation -> OK')

if __name__ == '__main__':
    main()
