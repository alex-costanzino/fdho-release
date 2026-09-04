import torch
import numpy as np
from torch.utils.data import Dataset

class SDFFitting(Dataset):
    def __init__(self, path, on_surface_samples, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud...")
        point_cloud = np.genfromtxt(path)
        print("Finished loading point cloud!")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high sample efficiency).
        coords -= np.mean(coords, axis = 0, keepdims=True)
        
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis = 0, keepdims=True)
            coord_min = np.amin(coords, axis = 0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_samples = on_surface_samples

        self.ch = 1

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_samples

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_samples
        total_samples = self.on_surface_samples + off_surface_samples

        # Sample random coordinates.
        rand_idcs = np.random.choice(point_cloud_size, size = self.on_surface_samples)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # Set on-surface = 0
        sdf[self.on_surface_samples:, :] = -1  # Set off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis = 0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis = 0)

        return {
            'coords': torch.from_numpy(coords).float(),
            'sdf': torch.from_numpy(sdf).float(), 
            'normals': torch.from_numpy(normals).float()
            }

class AnalyticSDFFitting(Dataset):
    def __init__(self, sdf_function, resolution=64):
        super().__init__()
        # Create a 3D grid of coordinates (N, N, N, 3)
        self.coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution),
            indexing='ij'
        ), dim=-1).view(-1, 3)
        
        # Calculate Ground Truth SDF values
        with torch.no_grad():
            self.values = sdf_function(self.coords)

        self.ch = 1
            
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        return self.coords, self.values

# Example Usage: Sphere
def sphere_sdf(coords):
    return torch.norm(coords, dim=1, keepdim=True) - 0.5