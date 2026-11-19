"""
Data Utilities
===============
Functions for locating the MRNet dataset root and building train/val/test DataFrames.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def find_mrnet_root(base_dir: str) -> str:
    """
    Walk base_dir to find the directory containing MRNet CSV label files.

    The expected directory structure is:
        <base_dir>/
            train-abnormal.csv
            train-acl.csv
            train-meniscus.csv
            valid-abnormal.csv
            valid-acl.csv
            valid-meniscus.csv
            train/
                sagittal/  (*.npy files)
                coronal/   (*.npy files)
                axial/     (*.npy files)
            valid/
                sagittal/
                coronal/
                axial/

    Args:
        base_dir: Root directory to search.

    Returns:
        Path to the MRNet root directory.

    Raises:
        FileNotFoundError: If the dataset cannot be located.
    """
    for root, _, files in os.walk(base_dir):
        if "train-abnormal.csv" in files:
            print(f"[INFO] Found MRNet dataset at: {root}")
            return root
    raise FileNotFoundError(
        f"Could not locate 'train-abnormal.csv' inside '{base_dir}'.\n"
        "Please ensure the MRNet dataset is correctly set up. "
        "See README.md → Dataset Setup for instructions."
    )


def build_dataframes(root_dir: str, val_fraction: float = 0.15, random_state: int = 42):
    """
    Build stratified train / internal-validation / held-out-test DataFrames.

    Stratification is performed across the joint (Abnormal, ACL, Meniscus)
    tuple to preserve class priors across splits (3-label multi-label
    stratification).

    The 'valid' split in MRNet is treated as the held-out test cohort
    (n=120), kept completely untouched during model development.
    The 'train' split is divided into training (85%) and internal
    validation (15%) sets.

    Args:
        root_dir:      MRNet root directory (from find_mrnet_root).
        val_fraction:  Fraction of training data to use for validation (default 0.15).
        random_state:  Random seed for reproducible splitting (default 42).

    Returns:
        (train_df, val_df, test_df): Three DataFrames with columns
            [case, abnormal, acl, meniscus, folder].
    """
    tasks = ["abnormal", "acl", "meniscus"]

    def load_split(prefix: str) -> pd.DataFrame:
        dfs = []
        for t in tasks:
            path = os.path.join(root_dir, f"{prefix}-{t}.csv")
            df = pd.read_csv(
                path,
                header=None,
                names=["case", t],
                dtype={"case": str, t: int},
            )
            df["case"] = df["case"].str.strip().str.zfill(4)
            dfs.append(df.set_index("case"))
        merged = pd.concat(dfs, axis=1).reset_index()
        merged["folder"] = prefix
        return merged

    full_train_df = load_split("train")
    test_df       = load_split("valid")

    # 3-label stratification key: "abn_acl_men" string tuple
    strat_key = (
        full_train_df["abnormal"].astype(str) + "_" +
        full_train_df["acl"].astype(str)      + "_" +
        full_train_df["meniscus"].astype(str)
    )
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=val_fraction,
        random_state=random_state,
        stratify=strat_key,
    )
    return train_df, val_df, test_df
