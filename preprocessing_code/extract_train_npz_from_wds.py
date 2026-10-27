"""
Prepare the EarthAlign training samples for the probing experiments.

The classification and cyclone physical quantity estimation probing scripts expect
the training and test samples as individual <folder_name>.npz files in the same
`npz` directory. The test samples are released as plain .npz files (inside the
npz-*.tar archives of `data_era5_processed`), while the training samples are only
released as WebDataset shards (`webdatasets_train`). This script extracts the
training .npz files from the shards and saves them under their sample names.

Steps:
  1. Download `EarthAlign/data_era5_processed` and `EarthAlign/webdatasets_train`
     from https://huggingface.co/datasets/MBZUAI/EarthAlign
  2. Extract the test archives into the `npz` directory, e.g.
       cd EarthAlign/data_era5_processed/npz && for f in npz-*.tar; do tar -xf "$f"; done
  3. Run this script with --npz-dir pointing to the same `npz` directory:
       python extract_train_npz_from_wds.py \
         --wds-dir   EarthAlign/webdatasets_train \
         --npz-dir   EarthAlign/data_era5_processed/npz \
         --info-json EarthAlign/data_era5_processed/info.json \
         --workers 8
     Requirements: numpy, tqdm. Disk space: ~200 GB for the training .npz files.

How the files are named:
  Inside the shards, samples are stored under an index (e.g. 000020991.npz), and the
  sample name is the `folder_name` field stored inside each .npz. A few names contain
  "/" (e.g. "... Flooding and/or mudflows"). Since "/" cannot be used in a file name,
  the corresponding files are named with "_" instead (e.g. "... Flooding and_or
  mudflows.npz"), which is the name used in info.json and in the probing config
  files. Some names also contain spaces; this is expected, so quote the paths if you
  handle these files from the shell.

Other notes:
  - The .npz files are copied as-is, without re-compression.
  - The script can be interrupted and re-run: files that already exist with the same
    size are skipped. Existing files with a different size are reported as conflicts
    and left untouched, unless --overwrite is given.
  - With --info-json, the script checks at the end that every training and test sample
    listed in info.json is present in the `npz` directory.
  - --dry-run reads the shards and reports the file names without writing any file.
  - The reports (summary, problems, renamed samples, full list) are saved in --report-dir.
"""

import io
import os
import csv
import glob
import json
import tarfile
import argparse
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

REPORT_FIELDNAMES = ["shard", "member", "folder_name", "saved_name", "sanitized", "status", "error"]


def sanitize_folder_name(folder_name):
    # "/" and "\" are path separators, so they are replaced to keep each sample a single file.
    return folder_name.replace("/", "_").replace("\\", "_")


def read_folder_name(npz_bytes):
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        if "folder_name" not in data.files:
            raise KeyError(f"'folder_name' not found (keys: {data.files})")

        folder_name = data["folder_name"]

    if folder_name.ndim != 0:
        raise ValueError(f"'folder_name' is not a scalar (shape={folder_name.shape})")

    folder_name = folder_name.item()
    if isinstance(folder_name, bytes):
        folder_name = folder_name.decode("utf-8")

    return str(folder_name)


def write_atomic(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(payload)
    os.replace(tmp_path, path)


def process_shard(args):
    shard_path, npz_dir, dry_run, overwrite = args
    shard = os.path.basename(shard_path)
    rows = []

    def add_row(member, folder_name="", saved_name="", sanitized=False, status="", error=""):
        rows.append({
            "shard": shard,
            "member": member,
            "folder_name": folder_name,
            "saved_name": saved_name,
            "sanitized": sanitized,
            "status": status,
            "error": error,
        })

    try:
        # Streaming mode: members are read sequentially, which is faster on network filesystems.
        with tarfile.open(shard_path, "r|") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".npz"):
                    continue

                try:
                    npz_bytes = tar.extractfile(member).read()
                    folder_name = read_folder_name(npz_bytes)
                    saved_name = sanitize_folder_name(folder_name) + ".npz"
                    sanitized = saved_name != folder_name + ".npz"

                    if folder_name.strip() in ("", ".", ".."):
                        raise ValueError(f"Invalid folder_name: {folder_name!r}")

                    out_path = os.path.join(npz_dir, saved_name)

                    if os.path.isfile(out_path) and not overwrite:
                        if os.path.getsize(out_path) == len(npz_bytes):
                            status = "skipped_exists"
                        else:
                            # A different file already exists: keep it and report it (use --overwrite to replace it).
                            status = "conflict_exists_different_size"
                    elif dry_run:
                        status = "dry_run"
                    else:
                        write_atomic(out_path, npz_bytes)
                        status = "written"

                    add_row(member.name, folder_name, saved_name, sanitized, status)

                except Exception as e:
                    add_row(member.name, status="error", error=f"{type(e).__name__}: {e}")

    except Exception as e:
        add_row("", status="shard_error", error=f"{type(e).__name__}: {e}")

    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def check_against_info_json(info_json, npz_dir, extracted_names):
    with open(info_json, "r", encoding="utf-8") as f:
        info = json.load(f)

    expected = {sample["ERA5_FOLDER_NAME"] + ".npz" for sample in info["samples"]}
    present = {filename for filename in os.listdir(npz_dir) if filename.endswith(".npz")}

    return {
        "expected": len(expected),
        "missing": sorted(expected - present),
        "extracted_not_in_info": sorted(set(extracted_names) - expected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wds-dir", required=True, help="Directory containing the training WebDataset .tar shards")
    parser.add_argument("--npz-dir", required=True, help="Output directory (the `npz` directory of the test data)")
    parser.add_argument("--info-json", default=None, help="Optional info.json to verify that all samples are present")
    parser.add_argument("--report-dir", default="wds_extract_report", help="Where to save the CSV/TXT reports")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="Number of shards processed in parallel")
    parser.add_argument("--dry-run", action="store_true", help="Read the shards, but do not write any .npz file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files")
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.wds_dir, "*.tar")))
    assert len(shard_paths) > 0, f"No .tar shards found in: {args.wds_dir}"

    os.makedirs(args.npz_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    print(f"Shards     : {len(shard_paths)} in {args.wds_dir}")
    print(f"Output dir : {args.npz_dir}")
    print(f"Dry run    : {args.dry_run}")

    rows = []
    tasks = [(shard_path, args.npz_dir, args.dry_run, args.overwrite) for shard_path in shard_paths]
    with Pool(args.workers) as pool:
        for shard_rows in tqdm(pool.imap_unordered(process_shard, tasks), total=len(tasks), desc="Shards", dynamic_ncols=True):
            rows.extend(shard_rows)

    rows.sort(key=lambda row: (row["shard"], row["member"]))

    # Two different samples must never map to the same output file.
    name_to_members = defaultdict(list)
    for row in rows:
        if row["saved_name"]:
            name_to_members[row["saved_name"]].append(f"{row['shard']}/{row['member']}")
    duplicates = {name: members for name, members in name_to_members.items() if len(members) > 1}

    status_counts = Counter(row["status"] for row in rows)
    num_samples = sum(row["member"] != "" for row in rows)
    num_sanitized = sum(row["sanitized"] is True for row in rows)

    summary_lines = [
        f"WebDataset dir                 : {args.wds_dir}",
        f"Output npz dir                 : {args.npz_dir}",
        f"Dry run                        : {args.dry_run}",
        f"Shards                         : {len(shard_paths)}",
        f"Samples (.npz) in shards       : {num_samples}",
        f"Names with '/' or '\\' replaced : {num_sanitized}",
        f"Duplicate output names         : {len(duplicates)}",
        "",
        "Status counts:",
    ] + [f"  {status:<32}: {count}" for status, count in sorted(status_counts.items())]

    info_check = None
    if args.info_json is not None and not args.dry_run:
        info_check = check_against_info_json(args.info_json, args.npz_dir, name_to_members.keys())
        summary_lines += [
            "",
            f"info.json check ({args.info_json}):",
            f"  Expected samples (train+test) : {info_check['expected']}",
            f"  Missing from npz dir          : {len(info_check['missing'])}",
            f"  Extracted but not in info.json: {len(info_check['extracted_not_in_info'])}",
        ]

    path_all = os.path.join(args.report_dir, "extract_all.csv")
    path_problems = os.path.join(args.report_dir, "extract_problems.csv")
    path_sanitized = os.path.join(args.report_dir, "extract_sanitized_names.csv")
    path_summary = os.path.join(args.report_dir, "extract_summary.txt")

    problem_statuses = {"error", "shard_error", "conflict_exists_different_size"}
    write_csv(path_all, rows)
    write_csv(path_problems, [row for row in rows if row["status"] in problem_statuses])
    write_csv(path_sanitized, [row for row in rows if row["sanitized"] is True])

    with open(path_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

        if duplicates:
            f.write("\nDuplicate output names:\n")
            for name, members in sorted(duplicates.items()):
                f.write(f"  {name}: {members}\n")

        if info_check is not None:
            for key in ["missing", "extracted_not_in_info"]:
                if info_check[key]:
                    f.write(f"\n{key}:\n")
                    f.writelines(f"  {name}\n" for name in info_check[key])

    print("\n" + "\n".join(summary_lines))
    print(f"\nReports saved in: {args.report_dir}")

    num_problems = sum(status_counts[status] for status in problem_statuses)
    if num_problems or duplicates or (info_check and info_check["missing"]):
        print("\nWARNING: some samples need attention, see the reports above.")
    elif args.dry_run:
        print("\nDry run finished. No problems found, no files were written.")
    else:
        print("\nDone. All samples extracted successfully.")


if __name__ == "__main__":
    main()
