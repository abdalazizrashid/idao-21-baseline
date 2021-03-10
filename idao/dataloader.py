import numpy as np
import torch
import os
import pathlib as path
from PIL import Image
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset


class IDAODataset(DatasetFolder):

    def name_to_energy(self, name):
        try:
            names = os.path.split(name)[-1].split("_")
            idx = [i for i, v in enumerate(names) if v == "keV"][0]
            return torch.tensor(float(names[idx - 1]))
        except Exception as e:
            return torch.tensor(-1.0)

    def name_to_index(self, name):
        return os.path.split(name)[-1].split('.')[0]

    def __getitem__(self, index: int):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.name_to_energy(path), self.name_to_index(path)

class InferenceDataset(Dataset):
    def __init__(self, main_dir, transform, loader=None):
        self.img_loaderj= img_loader
        self.main_dir = path.Path(main_dir)
        self.transform = transform
        self.all_imgs = list(self.main_dir.glob("*.png"))
        self.loader = loader

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = self.all_imgs[idx]
        image = self.loader(img_loc)
        tensor_image = self.transform(image)
        return tensor_image, img_loc.name

def img_loader(path: str):
    with Image.open(path) as img:
        img = np.array(img)
    return img
