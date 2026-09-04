from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor
from utils.utils import get_mgrid

class ImageFitting(Dataset):
    def __init__(self, path, size):
        super().__init__()

        img = Image.open(path).convert('RGB')
        transform = Compose([
            Resize(size),
            ToTensor(),
        ])
        img = transform(img) * 2.0 - 1.0
        
        self.ch, self.h, self.w = img.shape

        self.pixels = img.permute(1, 2, 0).view(-1, 3)
        self.coords = get_mgrid(img.shape[1], img.shape[2], 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels