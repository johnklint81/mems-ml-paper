from torch.utils.data import Dataset
from pathlib import Path
from itertools import chain
from PIL import Image
import numpy as np


class Ellipses(Dataset):
    def __init__(self, root, transform, complex_label=1, real_label=0):
        """
        Constructor
        Args:
            root (Path/str): Filepath to the data root, e.g. './train'
            transform (Compose): A composition of image transforms, see below.
        """
        root = Path(root)
        if not (root.exists() and root.is_dir()):
            raise ValueError(f"Data root '{root}' is invalid")

        self.root = root
        self.transform = transform
        self._complex_label = complex_label
        self._real_label = real_label
        self._samples = self._collect_samples()

    def __getitem__(self, index):
        """
        Get sample by index
        Args:
            index (int)
        Returns:
             The index'th sample (Tensor, int)
        """
        # Access the stored path and label for the correct index
        path, label = self._samples[index]
        # Load the image into memory
        img = Image.open(path)
        # Perform transforms, if any.
        if self.transform is not None:
            transform = np.random.choice(self.transform)
            img = transform(img)
        return img, label

    def __len__(self):
        """
        Total number of samples
        """
        # return len([name for name in os.listdir('.') if os.path.isfile(name)])
        return len(self._samples)

    def _collect_samples(self):
        """
        Collect all paths and labels
        Helper method for the constructor
        """
        complex_paths = self._collect_imgs_sub_dir(self.root / "complex")
        complex_paths_and_labels = map(lambda path: (path, self._complex_label), complex_paths)
        real_paths = self._collect_imgs_sub_dir(self.root / "real")
        real_paths_and_labels = map(lambda path: (path, self._real_label), real_paths)

        return sorted(list(chain(complex_paths_and_labels, real_paths_and_labels)), key=lambda x: x[0].stem)

    @staticmethod
    def _collect_imgs_sub_dir(sub_dir: Path):
        """
        Collect image paths in a directory
        Helper method for the constructor
        """
        if not sub_dir.exists():
            raise ValueError(f"Data root did not contain sub dir")
        return sub_dir.glob("*.png")

    def get_sample_by_id(self, id_):
        """
        Get sample by image id
        The indices do not correspond to the image id's in the filenames.
        Args:
            id_ (str): Image id, e.g. `real.321`
        """
        id_index = [path.stem for (path, _) in self._samples].index(id_)
        return self[id_index]
