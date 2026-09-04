import torch
import kornia
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor
from utils.utils import get_mgrid

class CTScanning(Dataset):
    def __init__(self, path, size, num_angles = 180):
        super().__init__()

        img = Image.open(path).convert('RGB')
        transform = Compose([
            Resize(size),
            ToTensor(),
        ])
        img = transform(img) * 2.0 - 1.0
        
        self.ch, self.h, self.w = img.shape
        self.gt_image = img

        self.coords = get_mgrid(self.h, self.w, 2)
        self.angles = torch.linspace(0, 180, num_angles)
        self.sinogram = self.radon(img.unsqueeze(0), self.angles)

    def radon(self, imten, angles):
        """
        Computes the Radon transform.
        imten: (1, C, H, W)
        angles: Tensor of angles in degrees.
        Returns: (C, Angles, W)
        """
        device = imten.device
        angles = angles.to(device)
        n_angles = len(angles)
        
        imten_rep = imten.repeat_interleave(n_angles, dim = 0)
        
        imten_rot = kornia.geometry.transform.rotate(
            imten_rep, 
            angles,
            padding_mode = 'zeros'
        )
        
        sinogram = imten_rot.sum(dim = 2).permute(1, 0, 2)
        
        return sinogram

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.sinogram, self.gt_image