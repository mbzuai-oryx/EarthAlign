from functools import partial
from itertools import islice
from typing import Callable, Optional, Sequence, Union

import tqdm
import torch

from .zero_shot_metadata import IBTRACS_WMO_WIND_TEMPLATES, IBTRACS_WMO_PRES_TEMPLATES
from .zero_shot_metadata import IBTRACS_WMO_WIND_VALUES, IBTRACS_WMO_PRES_VALUES

def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def build_zero_shot_regressor(
        model,
        tokenizer,
        values: Sequence[Union[int, float, str]],
        templates: Sequence[Union[Callable, str]],
        num_values_per_batch: Optional[int] = 64,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """Build zero-shot regressor weights by iterating over candidate values in batches.

    Args:
        model: CLIP model instance.
        tokenizer: CLIP tokenizer instance.
        values: A sequence of candidate regression values.
        templates: A sequence of callables or format() friendly strings to produce
            prompts per value.

            Callable templates should accept:
                template(value)

            String templates may use:
                template.format(value)
                template.format(value=value)
                template.format(v=value)

        num_values_per_batch: The number of values to batch together in each forward,
            all if None.
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.

    Returns:
        zeroshot_weights: Tensor with shape [embed_dim, num_values].
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(values, Sequence) and len(values) > 0

    num_templates = len(templates)
    num_values = len(values)

    if use_tqdm:
        import tqdm
        num_iter = 1 if num_values_per_batch is None else ((num_values - 1) // num_values_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_values_per_batch)
    else:
        iter_wrap = iter

    def _format_template(template, value):
        if callable(template):
            return template(value)

        try:
            return template.format(value=value, v=value)
        except (KeyError, IndexError):
            return template.format(value)

    def _process_batch(batch_values):
        num_batch_values = len(batch_values)

        texts = [
            _format_template(template, value)
            for value in batch_values
            for template in templates
        ]

        texts = tokenizer(texts).to(device)

        value_embeddings = model.encode_text(texts, normalize=True)
        value_embeddings = value_embeddings.reshape(
            num_batch_values,
            num_templates,
            -1,
        ).mean(dim=1)

        value_embeddings = value_embeddings / value_embeddings.norm(dim=1, keepdim=True)
        value_embeddings = value_embeddings.T

        return value_embeddings

    with torch.no_grad():
        if num_values_per_batch:
            batched_embeds = [
                _process_batch(batch)
                for batch in iter_wrap(batched(values, num_values_per_batch))
            ]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(values)

    return zeroshot_weights

def build_ibtracs_class_regressors(
        model,
        tokenizer,
        task,
        classnames,
        num_values_per_batch=64,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    assert task in ["wmo_wind", "wmo_pres"]

    dict_templates = {
        "wmo_wind": IBTRACS_WMO_WIND_TEMPLATES,
        "wmo_pres": IBTRACS_WMO_PRES_TEMPLATES,
    }[task]
    
    dict_values = {
        "wmo_wind": IBTRACS_WMO_WIND_VALUES,
        "wmo_pres": IBTRACS_WMO_PRES_VALUES,
    }[task]

    regressors = {}

    # print(f"\nBuilding zero-shot regressor for IBTrACS {task}.")
    # for classname in tqdm.tqdm(classnames, desc=f"Classes: {classnames}"):
    for classname in classnames:
        templates = dict_templates[classname]
        values    = dict_values[classname]

        regressor = build_zero_shot_regressor(
            model=model,
            tokenizer=tokenizer,
            values=values,
            templates=templates,
            num_values_per_batch=num_values_per_batch,
            device=device,
            use_tqdm=use_tqdm,
        )

        regressors[classname] = regressor

    return regressors, dict_values