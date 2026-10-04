import json
import os
import random

import webdataset as wds
from tqdm import tqdm

from utils import split_train_test

# ============================================================
# INPUT / OUTPUT PARAMETERS
# ============================================================

DIR_PROCESSED_FOLDER = "<PROCESSED_ERA5_DIR>" # output folder from 1_preprocess_to_npz_json.py
DIR_WEBDATASET_OUT = "<WEBDATASET_OUTPUT_DIR>"
MAX_SAMPLES_PER_SHARD = {"train": 10_000}

RANDOM_SEED = 42

INFO_JSON_PATH = os.path.join(DIR_PROCESSED_FOLDER, "info.json")
# ============================================================


def load_info_json():
    assert os.path.isfile(INFO_JSON_PATH), f"Missing info.json: {INFO_JSON_PATH}"

    with open(INFO_JSON_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)

    assert "samples" in info, "info.json must contain key: samples"
    assert "num_samples" in info, "info.json must contain key: num_samples"
    assert info["num_samples"] == len(info["samples"]), (
        f"num_samples mismatch: info says {info['num_samples']}, "
        f"but samples has length {len(info['samples'])}"
    )

    return info


def get_npz_path(folder_name):
    return os.path.join(DIR_PROCESSED_FOLDER, "npz", f"{folder_name}.npz")


def get_json_path(folder_name):
    return os.path.join(DIR_PROCESSED_FOLDER, "json", f"{folder_name}.json")


def check_intermediate_files_complete(samples):
    missing = []

    seen_folder_names = set()

    for sample in tqdm(samples):
        folder_name = sample["ERA5_FOLDER_NAME"]

        assert folder_name not in seen_folder_names, f"Duplicate ERA5_FOLDER_NAME: {folder_name}"
        seen_folder_names.add(folder_name)

        npz_path = get_npz_path(folder_name)
        json_path = get_json_path(folder_name)

        if not os.path.isfile(npz_path):
            missing.append(npz_path)

        if not os.path.isfile(json_path):
            missing.append(json_path)

    if missing:
        print(f"Found {len(missing)} missing intermediate files.")
        print("First missing files:")
        for path in missing[:20]:
            print(path)
        raise FileNotFoundError("Intermediate preprocessing is incomplete.")

    print(f"All intermediate files are complete: {len(samples)} samples")

def load_webdataset_sample(args):
    sample_idx, sample = args
    folder_name = sample["ERA5_FOLDER_NAME"]

    npz_path = get_npz_path(folder_name)
    json_path = get_json_path(folder_name)

    with open(npz_path, "rb") as f:
        npz_bytes = f.read()

    with open(json_path, "r", encoding="utf-8") as f:
        json_payload = json.load(f)

    return {
        "__key__": f"{sample_idx:09d}",
        "npz": npz_bytes,
        "json": json_payload,
    }

def write_webdataset_split(split_name, samples):
    output_dir = os.path.join(DIR_WEBDATASET_OUT, split_name)
    os.makedirs(output_dir, exist_ok=False)

    shard_pattern = os.path.join(output_dir, "%09d.tar")

    with wds.ShardWriter(
        shard_pattern,
        maxcount=MAX_SAMPLES_PER_SHARD[split_name],
    ) as sink:
        for sample_idx, sample in tqdm(
            enumerate(samples),
            total=len(samples),
            desc=f"Writing {split_name}",
            dynamic_ncols=True,
        ):
            out_sample = load_webdataset_sample((sample_idx, sample))
            sink.write(out_sample)

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    info = load_info_json()
    all_samples = info["samples"]

    check_intermediate_files_complete(all_samples)

    train_samples, test_samples = split_train_test(all_samples)

    print("\nTotal train and test sample size:")
    print(f"{len(train_samples)} | {len(test_samples)}")

    random.shuffle(train_samples)

    os.makedirs(DIR_WEBDATASET_OUT, exist_ok=False)

    write_webdataset_split("train", train_samples)

    print("\nDone.")