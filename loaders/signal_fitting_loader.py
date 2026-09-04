import torch
import numpy as np
from scipy.io import wavfile

class SignalFitting(torch.utils.data.Dataset):
    def __init__(self, function, num_samples, domain=(-1.0, 1.0)):
        """
        Args:
            function (callable): A function that takes a tensor of coords (N, 1) and returns a tensor of signal values (N, C).
            num_samples (int): The number of points to sample in the domain.
            domain (tuple): The (min, max) range for the input coordinates.
        """
        super().__init__()

        self.num_samples = num_samples
        
        # 1. Generate the input coordinates (similar to get_mgrid but 1D).
        # Shape: (num_samples, 1)
        self.coords = torch.linspace(domain[0], domain[1], steps=num_samples).float().view(-1, 1)

        # 2. Calculate the target values using the provided function.
        # Shape: (num_samples, Output_Dim)
        # We wrap it in torch.no_grad() to ensure we don't track gradients for data generation.
        with torch.no_grad():
            self.values = function(self.coords)

        # 3. Handle Normalization
        v_min = self.values.min()
        v_max = self.values.max()
                    
        # Avoid division by zero if the signal is a flat line.
        if v_max - v_min > 1e-6:
            self.values = 2.0 * (self.values - v_min) / (v_max - v_min) - 1.0
        else:
            # If signal is constant, just set it to 0.
            self.values = torch.zeros_like(self.values)

        # Ensure the output has at least 2 dimensions (Batch, Channels).
        if self.values.ndim == 1:
            self.values = self.values.view(-1, 1)
        
        self.ch = 1

        self.fs = num_samples / (domain[1] - domain[0])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: 
            raise IndexError
        
        return self.coords, self.values
    
class AudioFitting(torch.utils.data.Dataset):
    def __init__(self, file_path, domain=(-100.0, 100.0)):
        """
        Args:
            file_path (str): Path to the .wav file.
            domain (tuple): The (min, max) range for the input coordinates (time).
        """
        super().__init__()

        self.fs_raw, data = wavfile.read(file_path)
        
        if len(data.shape) > 1 and data.shape[1] == 2:
            data = np.mean(data, axis=1)
            
        data = data.astype(np.float32)

        self.num_samples = len(data)

        self.coords = torch.linspace(domain[0], domain[1], steps = self.num_samples).float().view(-1, 1)

        self.values = (torch.from_numpy(data) - 0.5) * 2

        if self.values.ndim == 1:
            self.values = self.values.view(-1, 1)

        self.ch = 1
        self.fs = self.num_samples / (domain[1] - domain[0])
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: 
            raise IndexError
        
        return self.coords, self.values
