"""
Convert the downloaded Stanford meshes into the oriented point clouds the SDF
loader expects.

`loaders.sdf_fitting_loader.SDFFitting` reads a whitespace-separated text file
with six columns per row: x y z nx ny nz. This script produces exactly that
from the .ply meshes fetched by scripts/download_sdf_data.sh.

Usage:
    python scripts/prepare_sdf_data.py                  # convert every .ply in data/sdf
    python scripts/prepare_sdf_data.py --shapes lucy    # convert one shape
    python scripts/prepare_sdf_data.py --max-points 5000000
"""

import argparse
import os
from glob import glob

import numpy as np
import trimesh

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sdf')


def convert(ply_path, out_path, max_points, seed):
    print(f"[{os.path.basename(ply_path)}] loading")
    mesh = trimesh.load(ply_path, process=False)

    if isinstance(mesh, trimesh.points.PointCloud):
        raise ValueError(f"{ply_path} holds a point cloud without faces; normals cannot be derived.")

    coords = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    if len(coords) != len(normals):
        raise ValueError(f"{ply_path}: {len(coords)} vertices but {len(normals)} normals.")

    if max_points is not None and len(coords) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords, normals = coords[idx], normals[idx]
        print(f"[{os.path.basename(ply_path)}] subsampled to {max_points} points")

    print(f"[{os.path.basename(ply_path)}] writing {len(coords)} points to {out_path}")
    np.savetxt(out_path, np.hstack([coords, normals]), fmt='%.6f')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--shapes', nargs='*', default=None,
                        help="Shape names to convert (default: every .ply in data/sdf).")
    parser.add_argument('--max-points', type=int, default=None,
                        help="Randomly subsample to at most this many surface points. "
                             "Lucy and the Thai statue are ~14M and ~5M points respectively; "
                             "subsampling keeps the .xyz files and their load times manageable.")
    parser.add_argument('--seed', type=int, default=7, help="Seed for subsampling.")
    parser.add_argument('--overwrite', action='store_true', help="Rewrite .xyz files that already exist.")
    args = parser.parse_args()

    if args.shapes:
        plys = [os.path.join(DATA_DIR, f'{s}.ply') for s in args.shapes]
    else:
        plys = sorted(glob(os.path.join(DATA_DIR, '*.ply')))

    if not plys:
        raise SystemExit(f"No .ply files in {DATA_DIR}. Run scripts/download_sdf_data.sh first.")

    for ply in plys:
        if not os.path.isfile(ply):
            raise SystemExit(f"Missing {ply}. Run scripts/download_sdf_data.sh first.")
        out = os.path.splitext(ply)[0] + '.xyz'
        if os.path.exists(out) and not args.overwrite:
            print(f"[{os.path.basename(ply)}] {out} already exists, skipping (use --overwrite).")
            continue
        convert(ply, out, args.max_points, args.seed)


if __name__ == '__main__':
    main()
