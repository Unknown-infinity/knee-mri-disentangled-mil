"""
MRNet Dataset Loaders
======================
Provides two dataset classes:
  - MRNetDataset      : V1 baseline (no augmentation, 256×256 raw slices)
  - MRNetDatasetV2    : V2 proposed (3D-consistent augmentation, 224×224 crops)

Both loaders read .npy volumetric files from the Stanford MRNet directory structure.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T


class MRNetDataset(Dataset):
    """V1 Baseline dataset – no spatial augmentation."""

    def __init__(self, root_dir: str, split_df, max_slices: int = 48):
        self.root_dir = root_dir
        self.df = split_df.reset_index(drop=True)
        self.max_slices = max_slices
        self.planes = ["sagittal", "coronal", "axial"]

    def __len__(self):
        return len(self.df)

    def _process_volume(self, vol: np.ndarray) -> torch.Tensor:
        s = vol.shape[0]
        if s > self.max_slices:
            start = (s - self.max_slices) // 2
            vol = vol[start : start + self.max_slices]

        vol = vol.astype(np.float32) / 255.0
        vol = np.stack([vol] * 3, axis=1)  # (S, 3, 256, 256)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        return torch.from_numpy((vol - mean) / std)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = str(row["case"]).zfill(4)
        folder  = row["folder"]

        volumes = {}
        for plane in self.planes:
            npy_path = os.path.join(self.root_dir, folder, plane, f"{case_id}.npy")
            vol = np.load(npy_path)
            volumes[plane] = self._process_volume(vol)

        labels = torch.tensor(
            [row["abnormal"], row["acl"], row["meniscus"]], dtype=torch.float32
        )
        return volumes["sagittal"], volumes["coronal"], volumes["axial"], labels


class MRNetDatasetV2(Dataset):
    """
    V2 Proposed dataset – 3D-consistent volumetric augmentation.

    Training augmentations applied identically across all S slices of each plane:
      • Random rotation ±12°
      • Random resized crop 224×224 (scale 0.82–1.0)
      • Multiplicative scanner-gain jitter U(0.9, 1.1)

    NOTE: Augmentation parameters are sampled independently per plane (sagittal,
    coronal, axial), which is anatomically appropriate because each plane is a
    different physical orientation. Within a single plane, all S slices receive
    the same spatial transform (true 3D consistency).
    """

    def __init__(self, root_dir: str, split_df, is_train: bool = True, max_slices: int = 44):
        self.root_dir   = root_dir
        self.df         = split_df.reset_index(drop=True)
        self.is_train   = is_train
        self.max_slices = max_slices
        self.planes     = ["sagittal", "coronal", "axial"]

        self.train_aug = T.Compose([
            T.RandomRotation(degrees=12),
            T.RandomResizedCrop(size=(224, 224), scale=(0.82, 1.0), antialias=True),
        ])
        self.eval_crop = T.CenterCrop(size=(224, 224))

    def __len__(self):
        return len(self.df)

    def _process_volume(self, vol: np.ndarray) -> torch.Tensor:
        s = vol.shape[0]
        if s > self.max_slices:
            start = (s - self.max_slices) // 2
            vol = vol[start : start + self.max_slices]

        vol = vol.astype(np.float32) / 255.0
        vol = np.stack([vol] * 3, axis=1)  # (S, 3, 256, 256)
        tensor_vol = torch.from_numpy(vol)

        # Apply 3D-consistent spatial transform across all S slices
        if self.is_train:
            tensor_vol = self.train_aug(tensor_vol)
            gain = torch.empty(1).uniform_(0.9, 1.1).item()
            tensor_vol = (tensor_vol * gain).clamp(0.0, 1.0)
        else:
            tensor_vol = self.eval_crop(tensor_vol)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor_vol - mean) / std

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = str(row["case"]).zfill(4)
        folder  = row["folder"]

        volumes = {}
        for plane in self.planes:
            npy_path = os.path.join(self.root_dir, folder, plane, f"{case_id}.npy")
            vol = np.load(npy_path)
            volumes[plane] = self._process_volume(vol)

        labels = torch.tensor(
            [row["abnormal"], row["acl"], row["meniscus"]], dtype=torch.float32
        )
        return volumes["sagittal"], volumes["coronal"], volumes["axial"], labels
