import argparse
import os
import torch
import torch.nn.functional as F
import json
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from open_clip import (
    create_model_and_transforms,
    get_tokenizer,
    build_ibtracs_class_regressors,
    IBTRACS_CLASSNAMES,
)

from open_clip_train.data import get_ibtracs_reg_dataset
from open_clip_train.zero_shot import compute_regression_metrics, run_regression


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--ibtracs-json-path", type=str, required=True)
    parser.add_argument("--ibtracs-npz-dir", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="amp")

    parser.add_argument("--plot-dir", type=str, required=True)

    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[len("module."):]] = v
        else:
            cleaned_state_dict[k] = v

    cleaned_state_dict = {key.replace(".vat", ".mettok"): val for key, val in cleaned_state_dict.items()}

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

def save_preds_vs_targets_plot(preds, targets, task, save_path="preds_vs_targets.png"):
    preds = preds.float().cpu()
    targets = targets.float().cpu()

    min_value = min(targets.min().item(), preds.min().item())
    max_value = max(targets.max().item(), preds.max().item())

    plt.figure(figsize=(7, 7))
    plt.scatter(targets.numpy(), preds.numpy(), alpha=0.5, s=18)

    # Perfect prediction line
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.xlabel(f"Target {task} speed")
    plt.ylabel(f"Predicted {task} speed")
    plt.title("Predictions vs Targets")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def print_metrics(metrics, title="Overall metrics"):
    print(f"\n{title}:")
    print(f"MAE       : {metrics['mae']:.4f}")
    print(f"RMSE      : {metrics['rmse']:.4f}")
    print(f"Bias      : {metrics['bias']:.4f}")
    print(f"Within 5  : {metrics['within_5']:.4f}")
    print(f"Within 10 : {metrics['within_10']:.4f}")
    print(f"R2        : {metrics['r2']:.4f}")
    print(f"Pearson   : {metrics['pearson']:.4f}")


def print_per_class_metrics(preds, targets, class_names, classnames, plot_dir, task):
    print("\nPer-class metrics:")
    print(f"{'class':25s} {'n':>8s} {'MAE':>10s} {'RMSE':>10s} {'Bias':>10s} {'R2':>10s} {'Pearson':>10s}")

    for classname in classnames:
        indices = [
            i for i, class_name in enumerate(class_names)
            if class_name == classname
        ]

        if len(indices) == 0:
            continue

        indices = torch.tensor(indices, dtype=torch.long)

        cls_preds = preds[indices]
        cls_targets = targets[indices]

        metrics = compute_regression_metrics(cls_preds, cls_targets)
        save_preds_vs_targets_plot(
            preds=cls_preds,
            targets=cls_targets,
            task=task,
            save_path=os.path.join(plot_dir, f"{classname}.png"),
        )

        print(
            f"{classname:25s} "
            f"{len(indices):8d} "
            f"{metrics['mae']:10.4f} "
            f"{metrics['rmse']:10.4f} "
            f"{metrics['bias']:10.4f} "
            f"{metrics['r2']:10.4f} "
            f"{metrics['pearson']:10.4f}"
        )


def main():
    args = parse_args()

    assert args.task in ["wmo_wind", "wmo_pres"]

    device = torch.device(args.device)

    model, _, _ = create_model_and_transforms(
        args.model,
        pretrained=None,
        device=device,
        precision=args.precision,
    )

    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    tokenizer = get_tokenizer(args.model)

    data = {}
    data["ibtracs-reg-val"] = get_ibtracs_reg_dataset(
        args.ibtracs_json_path,
        args.ibtracs_npz_dir,
        args.batch_size,
        args.workers,
        return_sample_name=True,
    )

    dataloader = data["ibtracs-reg-val"].dataloader

    classnames = IBTRACS_CLASSNAMES

    print(f"\nNumber of samples: {dataloader.num_samples}")
    print(f"Number of batches: {dataloader.num_batches}")
    print(f"Number of classes: {len(classnames)}")
    print(f"Classes: {classnames}")

    regressors, dict_values = build_ibtracs_class_regressors(
        model=model,
        tokenizer=tokenizer,
        task=args.task,
        classnames=classnames,
        device=args.device,
    )

    preds, targets, class_names, sample_names, similarities = run_regression(
        model=model,
        regressors=regressors,
        dict_values=dict_values,
        dataloader=dataloader,
        args=args,
        return_detail=True,
    )

    assert len(preds) == len(similarities)

    os.makedirs(args.plot_dir, exist_ok=True)

    print(f"\npreds shape: {preds.shape}")
    print(f"targets shape: {targets.shape}")

    metrics = compute_regression_metrics(preds, targets)
    print_metrics(metrics)
    save_preds_vs_targets_plot(
        preds=preds,
        targets=targets,
        task=args.task,
        save_path=os.path.join(args.plot_dir, "all.png"),
    )

    print_per_class_metrics(
        preds=preds,
        targets=targets,
        class_names=class_names,
        classnames=classnames,
        plot_dir=args.plot_dir,
        task=args.task,
    )

    absolute_errors = torch.abs(preds - targets)

    absolute_error_percentages = torch.where(
        targets != 0,
        absolute_errors / torch.abs(targets) * 100.0,
        torch.nan,
    )

    results = []

    for i in range(len(preds)):
        class_name = class_names[i]

        results.append({
            "sample_name": sample_names[i],
            "class_name": class_name,
            "target": targets[i].item(),
            "prediction": preds[i].item(),
            "absolute_error": absolute_errors[i].item(),
            "absolute_error_percentage": absolute_error_percentages[i].item(),
            "values": json.dumps(dict_values[class_name]),
            "similarities": json.dumps(
                similarities[i].squeeze(0).tolist()
            ),
        })

    df_results = pd.DataFrame(results)

    csv_path = os.path.join(
        args.plot_dir,
        "predictions.csv",
    )

    df_results.to_csv(csv_path, index=False)

    print(f"Saved prediction details to: {csv_path}")

if __name__ == "__main__":
    main()