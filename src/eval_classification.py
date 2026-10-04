import os
import argparse
import torch
import re
import json
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score

from meteolip import (
    create_model_and_transforms,
    get_tokenizer,
    get_input_dtype,
    build_zero_shot_classifier,
    NATURAL_EVENT_TEMPLATES,
)

from meteolip_train.data import get_classification_dataset
from meteolip_train.precision import get_autocast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cm-dir", type=str, required=True)
    parser.add_argument("--classification-json-path", type=str, required=True)
    parser.add_argument("--classification-npz-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="amp")
    parser.add_argument(
        "--save-detail-csv",
        action="store_true",
        help="Save per-sample classification logits and probabilities to CSV.",
    )
    parser.add_argument("--record-source", type=str, nargs="+", default=None)
    return parser.parse_args()


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


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


def run_and_collect(model, classifier, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    all_logits = []
    all_targets = []
    all_pred_classes = []
    if args.save_detail_csv:
        all_sample_names = []

    top1, top3, n = 0.0, 0.0, 0.0

    with torch.inference_mode():
        for batch in tqdm(dataloader, unit_scale=args.batch_size):
            if len(batch) == 2:
                images, target = batch
                if args.save_detail_csv:
                    raise ValueError(
                        "save_detail_csv=True, but the dataloader "
                        "does not return sample names."
                    )
            elif len(batch) == 3:
                images, target, sample_names = batch
            else:
                raise ValueError(f"len(batch) should be either 2 or 3, got {len(batch)}")

            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                output = model(image=images)
                image_features = output["image_features"] if isinstance(output, dict) else output[0]
                logits = model.logit_scale.exp() * (image_features @ classifier)

            pred_classes = logits.argmax(dim=-1)

            acc1, acc3 = accuracy(logits, target, topk=(1, 3))
            top1 += acc1
            top3 += acc3
            n += images.size(0)

            all_logits.append(logits.cpu())
            all_targets.append(target.cpu())
            all_pred_classes.append(pred_classes.cpu())
            if args.save_detail_csv:
                all_sample_names.extend(sample_names)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    pred_classes = torch.cat(all_pred_classes, dim=0)

    top1 = top1 / n
    top3 = top3 / n

    macro_f1 = f1_score(
        targets.numpy(),
        pred_classes.numpy(),
        average="macro",
        zero_division=0,
    )

    if args.save_detail_csv:
        return logits, targets, pred_classes, top1, top3, macro_f1, all_sample_names
    else:
        return logits, targets, pred_classes, top1, top3, macro_f1


def build_confusion_matrix(targets, preds, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets, preds):
        cm[t.long(), p.long()] += 1
    return cm


def print_per_class_accuracy(cm, class_names):
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        total = cm[i].sum().item()
        correct = cm[i, i].item()
        acc = (correct / total) if total > 0 else 0.0
        print(f"{class_name:25s}  {correct:4d}/{total:4d}  acc={acc:.4f}")

def save_confusion_matrix(cm, class_names, save_path="cls_weather_event_cm.png"):
    cm_np = cm.numpy()
    num_classes = len(class_names)

    # Save confusion matrix as CSV
    csv_save_path = os.path.splitext(save_path)[0] + ".csv"

    cm_df = pd.DataFrame(
        cm_np,
        index=class_names,
        columns=class_names,
    )

    cm_df.index.name = "true\\pred"
    cm_df.to_csv(csv_save_path)

    print(f"\nSaved confusion matrix CSV to {csv_save_path}")

    fig_w = max(10, 0.65 * num_classes)
    fig_h = max(9, 0.65 * num_classes)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Light blue confusion matrix style
    im = ax.imshow(
        cm_np,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=cm_np.max(),
        zorder=1,
    )

    # Axis labels and ticks
    ax.set_xlabel("Prediction", fontsize=20)
    ax.set_ylabel("Label", fontsize=20)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))

    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Put x-axis labels on top, like your example
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("bottom")

    # Thin white grid lines between cells
    ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.0,
        zorder=3,
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    # Bold solid orange diagonal highlight
    for i in range(num_classes):
        # Light orange fill overlay
        fill_rect = plt.Rectangle(
            (i - 0.5, i - 0.5),
            1,
            1,
            facecolor="#D55E00",
            edgecolor="none",
            alpha=0.12,
            zorder=4,
        )
        ax.add_patch(fill_rect)

        # Strong solid orange border
        border_rect = plt.Rectangle(
            (i - 0.5, i - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#D55E00",
            linewidth=3.0,
            linestyle="-",
            joinstyle="miter",
            zorder=6,
        )
        ax.add_patch(border_rect)

    # Annotate nonzero cells
    thresh = cm_np.max() * 0.45

    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_np[i, j]

            if value == 0:
                continue

            ax.text(
                j,
                i,
                f"{value}",
                ha="center",
                va="center",
                color="white" if value > thresh else "#1f4e89",
                fontsize=8,
                zorder=7,
            )

    # Black border around the whole matrix
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    # Keep matrix orientation correct
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(num_classes - 0.5, -0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved confusion matrix to {save_path}")

def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

def save_classification_details(
    logits,
    targets,
    pred_classes,
    sample_names,
    class_names,
    save_path,
):
    probabilities = torch.softmax(logits.float(), dim=-1)

    results = []

    for i in range(len(targets)):
        sorted_indices = torch.argsort(
            logits[i],
            descending=True,
        )

        sorted_indices_list = sorted_indices.tolist()

        sorted_values = [
            class_names[j]
            for j in sorted_indices_list
        ]

        sorted_logits = logits[i][sorted_indices].tolist()
        sorted_probabilities = probabilities[i][sorted_indices].tolist()

        target_idx = targets[i].item()
        pred_idx = pred_classes[i].item()

        results.append({
            "sample_name": sample_names[i],
            "target": target_idx,
            "target_class": class_names[target_idx],
            "prediction": pred_idx,
            "prediction_class": class_names[pred_idx],
            "values": json.dumps(sorted_values),
            "logits": json.dumps(sorted_logits),
            "probabilities": json.dumps(sorted_probabilities),
        })

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

    print(f"Saved classification details to: {save_path}")


def main():
    args = parse_args()
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

    dataloaders = get_classification_dataset(
        args.classification_json_path,
        args.classification_npz_dir,
        args.batch_size,
        args.workers,
        return_sample_name=args.save_detail_csv,
        record_source=args.record_source,
    )

    os.makedirs(args.cm_dir, exist_ok=True)

    autocast = get_autocast(args.precision, device_type=device.type)
    
    for classification_task, dataloader in dataloaders.items():
        print(f"\nStarting classification task: {classification_task}")
        
        print("Number of test samples:", len(dataloader.dataset))
        dataset_names = sorted({
            item[0].split("__", 1)[0]
            for item in dataloader.dataset.data
        })
        print("Dataset names:", dataset_names)

        if args.record_source is not None:
            assert set(dataset_names) == set(args.record_source), (
                f"Dataset names do not match record_source. "
                f"Found: {dataset_names}, expected: {args.record_source}"
            )

        if classification_task in ["natural_event", "cyclone", "temperature"]:
            class_templates = NATURAL_EVENT_TEMPLATES
        else:
            raise ValueError(f"Unknown classification task: {classification_task}")

        class_names = dataloader.dataset.classes

        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=class_names,
                templates=class_templates,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
            )

        outputs = run_and_collect(model, classifier, dataloader, args)
        if args.save_detail_csv:
            logits, targets, pred_classes, top1, top3, macro_f1, sample_names = outputs
        else:
            logits, targets, pred_classes, top1, top3, macro_f1 = outputs

        print(f"Task: {classification_task}")
        print(f"F1   : {macro_f1:.4f}")
        print(f"Top-1: {top1:.4f}")
        print(f"Top-3: {top3:.4f}")

        cm = build_confusion_matrix(
            targets,
            pred_classes,
            len(class_names),
        )

        print_per_class_accuracy(cm, class_names)

        safe_task_name = sanitize_filename(classification_task)
        cm_save_path = os.path.join(
            args.cm_dir,
            f"cls_{safe_task_name}_cm.png",
        )

        save_confusion_matrix(
            cm,
            class_names,
            save_path=cm_save_path,
        )

        print(f"Saved confusion matrix to: {cm_save_path}")
    
    if args.save_detail_csv:
        detail_save_path = os.path.join(
            args.cm_dir,
            f"cls_{safe_task_name}_details.csv",
        )

        save_classification_details(
            logits=logits,
            targets=targets,
            pred_classes=pred_classes,
            sample_names=sample_names,
            class_names=class_names,
            save_path=detail_save_path,
        )

if __name__ == "__main__":
    main()