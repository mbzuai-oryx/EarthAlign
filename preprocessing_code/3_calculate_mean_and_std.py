import io
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils_era5_vars import ALL_VARS

# ============================================================
# INPUT / OUTPUT PARAMETERS
# ============================================================

DIR_WEBDATASET = "<TRAIN_WEBDATASET_DIR>" # output folder from 2_create_wds_for_train_samples.py
NUM_WORKERS = 4
N_VARS = len(ALL_VARS)

# ============================================================


def process_npz_bytes(npz_bytes: bytes, source_name: str):
    with np.load(io.BytesIO(npz_bytes)) as npz:
        if "x" not in npz:
            raise ValueError(f"{source_name}: missing key 'x'")
        x = npz["x"]

    if x.ndim != 4:
        raise ValueError(f"{source_name}: expected x.shape = (T, C, H, W), got {x.shape}")

    T, C, H, W = x.shape

    if C != N_VARS:
        raise ValueError(f"{source_name}: expected C={N_VARS}, got C={C}")

    if H != W:
        raise ValueError(f"{source_name}: expected H == W, got H={H}, W={W}")

    x = x.astype(np.float64, copy=False)

    sum_x = x.sum(axis=(0, 2, 3))                # (C,)
    sum_x2 = np.square(x).sum(axis=(0, 2, 3))    # (C,)
    min_x = x.min(axis=(0, 2, 3))                # (C,)
    max_x = x.max(axis=(0, 2, 3))                # (C,)
    count = T * H * W

    return sum_x, sum_x2, min_x, max_x, count


def process_tar(tar_path: str):
    total_sum = np.zeros(N_VARS, dtype=np.float64)
    total_sum_sq = np.zeros(N_VARS, dtype=np.float64)
    total_min = np.full(N_VARS, np.inf, dtype=np.float64)
    total_max = np.full(N_VARS, -np.inf, dtype=np.float64)
    total_count = 0
    total_npz = 0

    with tarfile.open(tar_path, "r") as tar:
        for member in tqdm(tar):
            if not member.isfile() or not member.name.endswith(".npz"):
                continue

            f = tar.extractfile(member)
            if f is None:
                raise ValueError(f"{tar_path}: could not read {member.name}")

            sum_x, sum_x2, min_x, max_x, count = process_npz_bytes(
                f.read(), f"{tar_path}:{member.name}"
            )

            total_sum += sum_x
            total_sum_sq += sum_x2
            total_min = np.minimum(total_min, min_x)
            total_max = np.maximum(total_max, max_x)
            total_count += count
            total_npz += 1

    return total_sum, total_sum_sq, total_min, total_max, total_count, total_npz


def compute_stats(webdataset_dir: str, num_workers: int = 1):
    tar_files = sorted(Path(webdataset_dir).glob("*.tar"))
    if not tar_files:
        raise ValueError(f"No .tar files found in {webdataset_dir}")

    total_sum = np.zeros(N_VARS, dtype=np.float64)
    total_sum_sq = np.zeros(N_VARS, dtype=np.float64)
    total_min = np.full(N_VARS, np.inf, dtype=np.float64)
    total_max = np.full(N_VARS, -np.inf, dtype=np.float64)
    total_count = 0
    total_npz = 0

    if num_workers == 1:
        results = (
            process_tar(str(p))
            for p in tqdm(tar_files, desc="Processing tar files", unit="tar")
        )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(process_tar, str(p)) for p in tar_files]
            results = (
                f.result()
                for f in tqdm(as_completed(futures), total=len(futures), desc="Processing tar files", unit="tar")
            )

    for sum_x, sum_x2, min_x, max_x, count, n_npz in results:
        total_sum += sum_x
        total_sum_sq += sum_x2
        total_min = np.minimum(total_min, min_x)
        total_max = np.maximum(total_max, max_x)
        total_count += count
        total_npz += n_npz

    if total_count == 0:
        raise ValueError("No npz files found inside tar shards.")

    mean = total_sum / total_count
    var = total_sum_sq / total_count - np.square(mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    return mean, std, total_min, total_max, total_npz, total_count


if __name__ == "__main__":
    mean, std, min_val, max_val, total_npz, total_count = compute_stats(
        DIR_WEBDATASET,
        num_workers=NUM_WORKERS,
    )

    print(f"Processed {total_npz} npz files")
    print(f"Per-variable count: {total_count}")

    for i, name in enumerate(ALL_VARS):
        print(
            f"{i:02d} {name:>20s}  "
            f"mean={mean[i]:.8f}  std={std[i]:.8f}  "
            f"min={min_val[i]:.8f}  max={max_val[i]:.8f}"
        )

    np.savez(
        "era5_crop_stats.npz",
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        vars=np.array(ALL_VARS),
    )