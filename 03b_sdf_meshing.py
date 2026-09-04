import argparse
import json
import os
import time

import torch
import numpy as np
import yaml
import skimage.measure
import trimesh
from scipy.spatial import cKDTree

from loaders.utils import DotDict
from utils.common import make_output_dir, path_stem
from models.models_bank import get_model, model_names
from loaders.sdf_fitting_loader import SDFFitting


# ------------------------------------------------------------------ #
#  Mesh extraction                                                    #
# ------------------------------------------------------------------ #

def create_mesh(model, cfg, filename, resolution=512, batch_size=1_000_000,
                verbose=True):
    """Extract a mesh via Marching Cubes on a [-1, 1]³ grid."""
    if verbose:
        print(f"  Generating mesh grid at resolution {resolution}³...")

    grid_points = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, resolution),
        torch.linspace(-1, 1, resolution),
        torch.linspace(-1, 1, resolution),
        indexing='ij',
    ), dim=-1).reshape(-1, 3)

    model.eval()
    sdf_values = []
    total_inference_time = 0.0

    with torch.no_grad():
        for i in range(0, grid_points.shape[0], batch_size):
            coords = grid_points[i:i + batch_size].cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output, _ = model(coords)
            if cfg.net.model_name == 'bacon':
                output = output[-1]
            torch.cuda.synchronize()
            total_inference_time += time.perf_counter() - t0
            sdf_values.append(output.detach().cpu())

    sdf_values = torch.cat(sdf_values, dim=0).numpy().reshape(
        resolution, resolution, resolution
    )

    if verbose:
        print("  Running Marching Cubes...")

    voxel_size = 2.0 / (resolution - 1)
    try:
        verts, faces, normals, _ = skimage.measure.marching_cubes(
            sdf_values, level=0.0, spacing=[voxel_size] * 3
        )
    except ValueError:
        print("  Error: no zero-crossing found.")
        return None, total_inference_time

    verts += np.array([-1.0, -1.0, -1.0])

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals)
    mesh.export(filename)
    if verbose:
        print(f"  Saved mesh to {filename} "
              f"({len(verts):,} verts, {len(faces):,} faces).")

    return mesh, total_inference_time


# ------------------------------------------------------------------ #
#  Metrics                                                            #
# ------------------------------------------------------------------ #

def calculate_chamfer_distance(pred_mesh, gt_coords, num_samples=100_000):
    """Symmetric Chamfer Distance (mean squared)."""
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)

    if gt_coords.shape[0] > num_samples:
        idx = np.random.choice(gt_coords.shape[0], num_samples, replace=False)
        gt_sub = gt_coords[idx]
    else:
        gt_sub = gt_coords

    tree_gt = cKDTree(gt_sub)
    d_p2g, _ = tree_gt.query(pred_points)

    tree_pred = cKDTree(pred_points)
    d_g2p, _ = tree_pred.query(gt_sub)

    return float(np.mean(d_p2g ** 2) + np.mean(d_g2p ** 2))


def calculate_normal_consistency(pred_mesh, gt_coords, gt_normals,
                                  num_samples=100_000):
    """Mean |dot(pred_normal, gt_normal)| at corresponding surface points."""
    pred_points, face_idx = trimesh.sample.sample_surface(pred_mesh, num_samples)
    pred_normals = pred_mesh.face_normals[face_idx]

    tree_gt = cKDTree(gt_coords)
    _, nn_idx = tree_gt.query(pred_points)
    matched_gt_normals = gt_normals[nn_idx]

    dots = np.abs(np.sum(pred_normals * matched_gt_normals, axis=-1))
    return float(np.mean(dots))


def calculate_fscore(pred_mesh, gt_coords, num_samples=100_000, tau=0.01):
    """F-Score at threshold τ (fraction of bounding-box diagonal)."""
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)

    if gt_coords.shape[0] > num_samples:
        idx = np.random.choice(gt_coords.shape[0], num_samples, replace=False)
        gt_sub = gt_coords[idx]
    else:
        gt_sub = gt_coords

    bbox_diag = np.sqrt(3) * 2.0   # diagonal of [-1, 1]³
    threshold = tau * bbox_diag

    tree_gt = cKDTree(gt_sub)
    d_p2g, _ = tree_gt.query(pred_points)
    precision = float(np.mean(d_p2g < threshold))

    tree_pred = cKDTree(pred_points)
    d_g2p, _ = tree_pred.query(gt_sub)
    recall = float(np.mean(d_g2p < threshold))

    if precision + recall < 1e-8:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract and evaluate a mesh from a fitted SDF.")
    parser.add_argument("--config", default="configs/config_sdf_fitting.yaml",
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        cfg = DotDict(**conf)

    out_dir = make_output_dir(path_stem(cfg.data.sdf_path), "sdf_fitting", create=False)

    if not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"Output directory '{out_dir}' not found. "
            f"Run 03a_sdf_fitting.py first."
        )

    dummy_sdf = SDFFitting(cfg.data.sdf_path, 1, 1)
    gt_coords = np.array(dummy_sdf.coords).copy()
    gt_normals = np.array(dummy_sdf.normals).copy()

    logs_path = os.path.join(out_dir, "logs.json")
    if os.path.exists(logs_path):
        with open(logs_path) as f:
            logs = json.load(f)
    else:
        logs = {}

    for model_name in model_names:

        print(f'Meshing with {model_name}...')
        cfg.net.model_name = model_name

        model_dir = os.path.join(out_dir, model_name)
        checkpoint_path = os.path.join(model_dir, "final_model.pt")

        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint not found at {checkpoint_path}, skipping.")
            continue

        # ---- Load model ---- #
        model, _, _ = get_model(cfg, dummy_sdf)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.cuda()
        print(f"  Loaded weights from {checkpoint_path}.")

        # ---- Extract mesh ---- #
        mesh_path = os.path.join(model_dir, "reconstruction.ply")
        mesh, mesh_inference_time = create_mesh(
            model, cfg, mesh_path,
            resolution=cfg.data.resolution,
        )

        if mesh is None:
            print(f"  Mesh extraction failed for {model_name}.")
            continue

        # ---- Metrics ---- #
        chamfer = calculate_chamfer_distance(mesh, gt_coords)
        print(f"  Chamfer Distance:     {chamfer:.6f}")

        nc = calculate_normal_consistency(mesh, gt_coords, gt_normals)
        print(f"  Normal Consistency:   {nc:.4f}")

        fscore = calculate_fscore(mesh, gt_coords, tau=0.01)
        print(f"  F-Score (τ=1%):       {fscore:.4f}")

        # ---- Update logs ---- #
        if model_name not in logs:
            logs[model_name] = {}

        logs[model_name].update({
            'chamfer_distance': chamfer,
            'normal_consistency': nc,
            'f_score': fscore,
            'mesh_inference_time': mesh_inference_time,
            'mesh_vertices': len(mesh.vertices),
            'mesh_faces': len(mesh.faces),
        })

        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=4)

    print(f"\nAll meshing metrics saved to {logs_path}.")