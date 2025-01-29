import pandas as pd
from itertools import zip_longest
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np


class Ellipses(Dataset):
    def __init__(self, img_root, data_root, transform, y_train_max):
        img_root = Path(img_root)
        data_root = Path(data_root)
        if not (img_root.exists() and img_root.is_dir()):
            raise ValueError(f"Image root '{img_root}' is invalid")
        self.img_root = img_root
        if not (data_root.exists() and data_root.is_dir()):
            raise ValueError(f"Data root '{data_root}' is invalid")
        self.y_train_max = y_train_max
        self.img_root = img_root
        self.data_root = data_root
        self.transform = transform
        self._samples = self._collect_samples()

    def __getitem__(self, index):
        img_path, data_path = self._samples[index]
        img = Image.open(img_path)
        data = pd.read_csv(data_path, header=None, skiprows=1)
        eigs = data[0].to_numpy()
        qfs = data[1].to_numpy()
        data = (np.append(eigs, qfs) / self.y_train_max).to_numpy()
        if self.transform is not None:
            transform = np.random.choice(self.transform)
            img = transform(img)
        return img, data

    def __len__(self):
        return len(self._samples)

    def _collect_samples(self):
        img_paths = self._collect_imgs_sub_dir(self.img_root)
        img_paths_and_labels = sorted(map(lambda path: path, img_paths))
        data_paths = self._collect_data_sub_dir(self.data_root)
        data_and_labels = sorted(map(lambda path: path, data_paths))
        return sorted(list(zip_longest(img_paths_and_labels, data_and_labels)))

    @staticmethod
    def _collect_imgs_sub_dir(sub_dir: Path):
        if not sub_dir.exists():
            raise ValueError(f"Image root did not contain sub dir")
        return sub_dir.glob("*.png")

    @staticmethod
    def _collect_data_sub_dir(sub_dir: Path):
        if not sub_dir.exists():
            raise ValueError(f"Data root did not contain sub dir")
        return sub_dir.glob("*.csv")
