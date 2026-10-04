import logging
import math

import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import f1_score

from meteolip import get_input_dtype, get_tokenizer, build_zero_shot_classifier
from meteolip import build_ibtracs_class_regressors
from meteolip import NATURAL_EVENT_TEMPLATES, IBTRACS_CLASSNAMES
from meteolip_train.precision import get_autocast

def round_to_nearest_odd(x: float) -> int:
    lower = math.floor(x)
    upper = math.ceil(x)

    # Make lower odd
    if lower % 2 == 0:
        lower -= 1

    # Make upper odd
    if upper % 2 == 0:
        upper += 1

    # Choose whichever odd number is closer
    if abs(x - lower) <= abs(x - upper):
        return lower
    else:
        return upper

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def run_classification(model, classifier, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    all_targets = []
    all_pred_classes = []

    with torch.inference_mode():
        top1, top3, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            pred_classes = logits.argmax(dim=-1)

            # Collect predictions and targets for macro F1
            all_targets.append(target.cpu())
            all_pred_classes.append(pred_classes.cpu())

            # measure accuracy
            acc1, acc3 = accuracy(logits, target, topk=(1, 3))
            top1 += acc1
            top3 += acc3
            n += images.size(0)

    top1 = (top1 / n)
    top3 = (top3 / n)

    targets = torch.cat(all_targets).numpy()
    pred_classes = torch.cat(all_pred_classes).numpy()

    macro_f1 = f1_score(
        targets,
        pred_classes,
        average="macro",
        zero_division=0,
    )

    return top1, top3, macro_f1

def compute_regression_metrics(preds, targets):
    preds = preds.float()
    targets = targets.float()

    errors = preds - targets
    abs_errors = errors.abs()

    mae = F.l1_loss(preds, targets).item()
    rmse = torch.sqrt(F.mse_loss(preds, targets)).item()

    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")

    pearson = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": errors.mean().item(),
        "within_5": (abs_errors <= 5).float().mean().item(),
        "within_10": (abs_errors <= 10).float().mean().item(),
        "r2": r2,
        "pearson": pearson,
    }

def median_index_from_topk(logits: torch.Tensor, top_k: int = 10) -> torch.Tensor:
    """
    Selects the median index among the top-k highest-scoring logits.

    This matches the logic from:
        find_median_from_top_n(text_dot_prods, n)

    Steps:
    1. Find the top-k scoring indices.
    2. Sort those indices by their original order.
    3. Pick the median index.

    Args:
        logits: Tensor of shape [num_values] or [batch_size, num_values].
        top_k: Number of top-scoring candidates to consider.

    Returns:
        Tensor of selected indices.
        Shape is [] if input was [num_values].
        Shape is [batch_size] if input was [batch_size, num_values].
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    squeeze_output = False

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    elif logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [num_values] or [batch_size, num_values], "
            f"got shape {tuple(logits.shape)}"
        )

    num_values = logits.shape[-1]
    k = min(top_k, num_values)

    topk_indices = torch.topk(logits, k=k, dim=-1).indices
    topk_indices_sorted = torch.sort(topk_indices, dim=-1).values

    median_indices = topk_indices_sorted[:, k // 2]

    if squeeze_output:
        median_indices = median_indices.squeeze(0)

    return median_indices

def run_regression(model, regressors, dict_values, dataloader, args, return_detail=False):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    dict_values_tensor = {
        classname: torch.tensor(values, device=device, dtype=torch.float32)
        for classname, values in dict_values.items()
    }

    all_preds = []
    all_targets = []
    all_class_names = []
    if return_detail:
        all_sample_names = []
    all_similarities = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 3:
                images, class_names, targets = batch
                if return_detail:
                    raise ValueError(
                        "return_detail=True, but the dataloader "
                        "does not return sample names."
                    )
            elif len(batch) == 4:
                images, class_names, targets, sample_names = batch
            else:
                raise ValueError(f"len(batch) should be either 3 or 4, got {len(batch)}")
            images = images.to(device=device, dtype=input_dtype)
            targets = targets.to(device=device, dtype=torch.float32)

            with autocast():
                output = model(image=images)

                image_features = output["image_features"] if isinstance(output, dict) else output[0]
                image_features = F.normalize(image_features, dim=-1)

            batch_preds = []

            for i, class_name in enumerate(class_names):
                regressor = regressors[class_name].to(
                    device=device,
                    dtype=image_features.dtype,
                )

                logits = model.logit_scale.exp() * (image_features[i:i + 1] @ regressor)

                assert len(dict_values_tensor[class_name]) == logits.shape[-1], (
                    f"Mismatch between number of values and logits for class '{class_name}': "
                    f"{len(dict_values_tensor[class_name])} values vs {logits.shape[-1]} logits."
                )

                top_k = round_to_nearest_odd(logits.shape[-1] * 0.05)
                top_k = max(1, min(top_k, logits.shape[-1]))
                if top_k % 2 == 0: top_k -= 1
                pred_index = median_index_from_topk(logits, top_k=top_k)
                pred_value = dict_values_tensor[class_name][pred_index]

                batch_preds.append(pred_value)
                all_similarities.append(logits.cpu())

            batch_preds = torch.cat(batch_preds, dim=0)

            all_preds.append(batch_preds.cpu())
            all_targets.append(targets.cpu())
            all_class_names.extend(class_names)
            if return_detail: all_sample_names.extend(sample_names)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if return_detail:
        return preds, targets, all_class_names, all_sample_names, all_similarities
    else:
        return preds, targets, all_class_names

def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    list_zero_shot_val_task = [
        "classification-val",
        "ibtracs-wind-val",
        "ibtracs-pres-val",
    ]

    if not any(task in data for task in list_zero_shot_val_task):
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod and hasattr(model, 'module'):
        model = model.module

    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    results = {}
    if 'classification-val' in data:
        for classification_task, classification_dataloader in data['classification-val'].items():
            logging.info(f'Starting zero-shot classification: {classification_task} ...')

            if classification_task in ["natural_event"]:
                class_templates = NATURAL_EVENT_TEMPLATES
            else:
                raise ValueError(f"Unknown classification task: {classification_task}")

            device = torch.device(args.device)
            autocast = get_autocast(args.precision, device_type=device.type)

            with autocast():
                classifier = build_zero_shot_classifier(
                    model,
                    tokenizer=tokenizer,
                    classnames=classification_dataloader.dataset.classes,
                    templates=class_templates,
                    num_classes_per_batch=10,
                    device=device,
                    use_tqdm=True,
                )
            
            top1, top3, macro_f1 = run_classification(model, classifier, classification_dataloader, args)
            results[f'cls-{classification_task}-val-f1']   = macro_f1
            results[f'cls-{classification_task}-val-top1'] = top1
            results[f'cls-{classification_task}-val-top3'] = top3

            logging.info(f'Finished zero-shot classification: {classification_task} ...')
    
    if 'ibtracs-wind-val' in data:
        logging.info('Starting zero-shot IBTrACS WMO_WIND regression.')

        autocast = get_autocast(args.precision, device_type=device.type)

        with autocast():
            regressors, dict_values = build_ibtracs_class_regressors(
                model=model,
                tokenizer=tokenizer,
                task="wmo_wind",
                classnames=IBTRACS_CLASSNAMES,
                device=args.device,
            )

        preds, targets, class_names = run_regression(
            model=model,
            regressors=regressors,
            dict_values=dict_values,
            dataloader=data["ibtracs-wind-val"].dataloader,
            args=args,
        )

        metrics = compute_regression_metrics(preds, targets)

        results['reg-ibtracs-wind-pearson'] = metrics['pearson']
        results['reg-ibtracs-wind-mae'] = metrics['mae']
        results['reg-ibtracs-wind-rmse'] = metrics['rmse']
        results['reg-ibtracs-wind-r2'] = metrics['r2']

        logging.info('Finished zero-shot IBTrACS WMO_WIND regression.')
    
    if 'ibtracs-pres-val' in data:
        logging.info('Starting zero-shot IBTrACS WMO_PRES regression.')

        autocast = get_autocast(args.precision, device_type=device.type)

        with autocast():
            regressors, dict_values = build_ibtracs_class_regressors(
                model=model,
                tokenizer=tokenizer,
                task="wmo_pres",
                classnames=IBTRACS_CLASSNAMES,
                device=args.device,
            )

        preds, targets, class_names = run_regression(
            model=model,
            regressors=regressors,
            dict_values=dict_values,
            dataloader=data["ibtracs-pres-val"].dataloader,
            args=args,
        )

        metrics = compute_regression_metrics(preds, targets)

        results['reg-ibtracs-pres-pearson'] = metrics['pearson']
        results['reg-ibtracs-pres-mae'] = metrics['mae']
        results['reg-ibtracs-pres-rmse'] = metrics['rmse']
        results['reg-ibtracs-pres-r2'] = metrics['r2']

        logging.info('Finished zero-shot IBTrACS WMO_PRES regression.')

    return results
