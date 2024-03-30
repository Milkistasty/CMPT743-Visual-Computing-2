import os

import numpy as np
import trimesh
from scipy.spatial import KDTree
from skimage import measure
import torch
from model import *

def generate_grid(point_cloud, res):
    """Generate grid over the point cloud with given resolution
    Args:
        point_cloud (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
    Returns:
        coords (np.array, [res*res*res, 3]): grid vertices
        coords_matrix (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]
    """
    b_min = np.min(point_cloud, axis=0)
    b_max = np.max(point_cloud, axis=0)

    coords = np.mgrid[:res, :res, :res]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    length += length / res
    coords_matrix[0, 0] = length[0] / res
    coords_matrix[1, 1] = length[1] / res
    coords_matrix[2, 2] = length[2] / res
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    coords = coords.T

    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples):
    """Predict occupancy of values batch-wise
    Args:
        points (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_samples (int): number of points to evaluate at once
    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """

    num_pts = points.shape[0]
    occ = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        occ[i * num_samples : i * num_samples + num_samples] = eval_func(
            points[i * num_samples : i * num_samples + num_samples]
        ).detach().cpu().numpy().squeeze()
    if num_pts % num_samples:
        occ[num_batches * num_samples :] = eval_func(
            points[num_batches * num_samples :]
        ).detach().cpu().numpy().squeeze()

    return occ

"""
the original one
"""
def eval_grid(coords, eval_func, num_per_sample=1024):
    """Predict occupancy of values on a grid
    Args:
        coords (np.array, [N, 3]): 3D coordinates of N points in space
        eval_func (function): function that takes a batch of points and returns occupancy values
        num_per_sample (int): number of points to evaluate at once

    Returns:
        occ (np.array, [N,]): occupancy values for each point
    """
    coords = coords.reshape([-1, 3])
    occ = batch_eval(coords, eval_func, num_samples=num_per_sample)
    return occ


def reconstruct(model, grid, res, transform):
    """Reconstruct mesh by predicting occupancy values on a grid
    Args:
        model (function): function that takes a batch of points and returns occupancy values
        grid (np.array, [N, 3]): 3D coordinates of N points in space
        res (int): grid resolution
        transform (np.array, [4, 4]): transform matrix: [0,res]x[0,res]x[0,res] -> [x_min, x_max]x[y_min, y_max]x[z_min, z_max]

    Returns:
        verts (np.array, [M, 3]): 3D coordinates of M vertices
        faces (np.array, [K, 3]): indices of K faces
    """
    
    occ = eval_grid(grid, model)
    occ = occ.reshape([res, res, res])

    # print("Min occupancy:", occ.min())
    # print("Max occupancy:", occ.max())

    # 0 if the output of model in range [-1, 1]; 0.5 if the output of model in range [0, 1]
    # if occ.min() >= 0 and occ.max() <= 1:
    #     surface_level = 0.5
    # else:
    #     surface_level = 0

    verts, faces, normals, values = measure.marching_cubes(occ, 0.5)
    verts = np.matmul(transform[:3, :3], verts.T) + transform[:3, 3:4]
    verts = verts.T

    return verts, faces


def compute_metrics(reconstr_path, gt_path, num_samples=1000000):
    """Compute chamfer and hausdorff distances between the reconstructed mesh and the ground truth mesh
    Args:
        reconstr_path (str): path to the reconstructed mesh
        gt_path (str): path to the ground truth mesh
        num_samples (int): number of points to sample from each mesh

    Returns:
        chamfer_dist (float): chamfer distance between the two meshes
        hausdorff_dist (float): hausdorff distance between the two meshes
    """
    reconstr = trimesh.load(reconstr_path)
    gt = trimesh.load(gt_path)

    # sample points on the mesh surfaces using trimesh
    reconstr_pts = reconstr.sample(num_samples)
    gt_pts = gt.sample(num_samples)

    # compute chamfer distance between the two point clouds
    reconstr_tree = KDTree(reconstr_pts)
    gt_tree = KDTree(gt_pts)
    dist1, _ = reconstr_tree.query(gt_pts)
    dist2, _ = gt_tree.query(reconstr_pts)
    chamfer_dist = (dist1.mean() + dist2.mean()) / 2
    hausdorff_dist = max(dist1.max(), dist2.max())

    return chamfer_dist, hausdorff_dist


"""
not working, only the ear of the rabbit will be kept
"""
def get_largest_volume_object(rec_verts, rec_faces, volume_threshold=0.1):
    """
    Filters out low volume objects and returns the largest volume object.

    Args:
        rec_verts (np.array, [M, 3]): Reconstructed mesh vertices.
        rec_faces (np.array, [K, 3]): Reconstructed mesh faces (triangle indices).
        volume_threshold (float): Volume threshold below which connected components are removed.

    Returns:
        trimesh.Trimesh: Largest volume object from the reconstructed mesh.
    """
    mesh = trimesh.Trimesh(vertices=rec_verts, faces=rec_faces)

    # Compute volumes of connected components
    components = mesh.split()

    # Filter components based on volume threshold
    large_components = [comp for comp in components if comp.volume >= volume_threshold]

    # If there are no large components, return the entire mesh
    if not large_components:
        return mesh

    # Find the largest component among the large components
    largest_component = max(large_components, key=lambda comp: comp.volume)

    return largest_component



if __name__ == "__main__":
    for model in os.listdir("model"):
        if model.endswith(".pth"):
            model_path = os.path.join("./model/", model)

            print(f"Evaluating model for {model} ...")

            filename = os.path.basename(model)
            object_name = filename.split("_OCCNet")[0] + ".obj"
            
            # Single LoD
            # model = OCCNet(grid_type="dense", grid_feat_dim=16, base_lod=8, num_lods=1, mlp_hidden_dim=64, num_layers=2)
            # Multi LoD
            # model = OCCNet(grid_type="dense", grid_feat_dim=16, base_lod=6, num_lods=3, mlp_hidden_dim=64, num_layers=2)
            # Hash
            model = OCCNet(grid_type="hash", grid_feat_dim=4, base_lod=4, num_lods=6, mlp_hidden_dim=256, num_layers=9)

            model.load_state_dict(torch.load(model_path))
            model.eval() 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            # ************************************************************

            for resolution in [64, 128, 256]:
                print("Resolution for Evaluation:", resolution)
                if object_name in os.listdir("processed"):
                    pc = trimesh.load(f"processed/{object_name}")
                    verts = np.array(pc.vertices)
                    # gt_occ = np.array(pc.visual.vertex_colors)[:, 0]
                    # gt_occ = (gt_occ == 0).astype("float32") * -2 + 1
                    # model = Baseline(verts, gt_occ)

                    # resolution = 64
                    grid, transform = generate_grid(verts, res=resolution)
                    rec_verts, rec_faces = reconstruct(model, grid, resolution, transform)

                    reconstr_path = f"reconstructions/{object_name}"
                    os.makedirs(os.path.dirname(reconstr_path), exist_ok=True)

                    trimesh.Trimesh(rec_verts, rec_faces).export(reconstr_path)
                    
                    #TODO: get the max volumes object only; not working, because the bodies of the rabbit are not connected together
                    # largest_object = get_largest_volume_object(rec_verts, rec_faces, volume_threshold=-0.1)
                    # largest_object.export(reconstr_path)

                    gt_path = f"data/{object_name}"

                    chamfer_dist, hausdorff_dist = compute_metrics(
                        reconstr_path, gt_path, num_samples=100000
                    )

                    print(object_name)
                    print(f"Chamfer distance: {chamfer_dist:.4f}")
                    print(f"Hausdorff distance: {hausdorff_dist:.4f}")
                    print("##################")
