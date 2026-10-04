import ast
import copy
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from meteolip.utils_climate import ERA5_CROP_STAT, CLIMATE_INJECT_POSITION
from meteolip.utils_climate import CLIMATE_INJECT_TIME_HOUR, CLIMATE_INJECT_TIME_DOY, CLIMATE_INJECT_TIME_YEAR

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('json' in sample)
    has_image = ('npz' in sample)
    return has_caption and has_image

def filter_no_label_or_no_image(sample):
    has_label = ('txt' in sample)
    has_image = ('npz' in sample)
    return has_label and has_image

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(
        args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None,
        caption_style=None, return_retrieval_metadata=False,
    ):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.rename(image="npz", text="json"),
    ])

    if return_retrieval_metadata:
        pipeline.extend([
            wds.map(load_npz_and_process_ret),
            wds.map(load_json_and_process_ret(tokenizer, caption_style)),
            wds.to_tuple("image", "text", "folder_name", "raw_text"),
        ])
    else:
        pipeline.extend([
            wds.map(load_npz_and_process),
            wds.map(load_json_and_process(tokenizer, caption_style)),
            wds.to_tuple("image", "text"),
        ])

    pipeline.extend([
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_retrieval_val_dataset(
    json_path, npz_dir, cap_dir, caption_style, return_metadata,
    batch_size, workers, tokenizer, record_source=None,
    direct_data_json=None,
):
    class RetrievalDataset(torch.utils.data.Dataset):
        def __init__(
            self, json_path, npz_dir, cap_dir, caption_style,
            return_metadata, record_source=None, direct_data_json=None,
        ):
            if json_path is not None and direct_data_json is not None:
                raise ValueError("Only one of 'json_path' or 'direct_data_json' can be provided, not both.")
            if json_path is None and direct_data_json is None:
                raise ValueError("Either 'json_path' or 'direct_data_json' must be provided.")
            assert os.path.isdir(npz_dir), f"NPZ directory not found: {npz_dir}"
            assert os.path.isdir(cap_dir), f"Caption directory not found: {cap_dir}"
            
            self.npz_dir = npz_dir
            self.cap_dir = cap_dir

            if json_path is not None:
                assert os.path.isfile(json_path), f"JSON file not found: {json_path}"
                with open(json_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            else:
                assert isinstance(direct_data_json, list), (f"direct_data_json must be a list, got {type(direct_data_json)}")
                self.data = copy.deepcopy(direct_data_json)
            
            if record_source is not None:
                assert isinstance(record_source, list)
                assert all(isinstance(source, str) for source in record_source)

                self.data = [
                    item for item in self.data
                    if any(item[0].startswith(f"{source}__") for source in record_source)
                ]

            self.return_metadata = return_metadata
            self.load_json_and_process = load_json_and_process(tokenizer, caption_style, return_raw_text=True)
        
        def __len__(self):
            return len(self.data)

        def _get_npz_path(self, folder_name):
            npz_path = os.path.join(self.npz_dir, f"{folder_name}.npz")
            assert os.path.isfile(npz_path), f"NPZ file not found: {npz_path}"
            return npz_path

        def _get_cap_path(self, folder_name):
            cap_path = os.path.join(self.cap_dir, f"{folder_name}.json")
            assert os.path.isfile(cap_path), f"Caption file not found: {cap_path}"
            return cap_path

        def __getitem__(self, index):
            folder_name, years = self.data[index]
            
            with open(self._get_npz_path(folder_name), "rb") as f:
                npz_bytes = f.read()
            image = load_npz_and_process({"image": npz_bytes})["image"]
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)

            with open(self._get_cap_path(folder_name), "rb") as f:
                cap_bytes = f.read()
            json_out = self.load_json_and_process({"text": cap_bytes})
            text, raw_text = json_out["text"], json_out["raw_text"]
            if not isinstance(text, torch.Tensor):
                text = torch.as_tensor(text)

            if self.return_metadata:
                return image, text, folder_name, raw_text
            else:
                return image, text

    dataset = RetrievalDataset(
        json_path, npz_dir, cap_dir, caption_style, return_metadata,
        record_source, direct_data_json,
    )
    sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )

    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_classification_dataset(
    json_path, npz_dir, batch_size, workers,
    return_sample_name=False, record_source=None,
):
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, task_data, npz_dir, return_sample_name=False, record_source=None):
            assert os.path.isdir(npz_dir), f"NPZ directory not found: {npz_dir}"

            self.npz_dir = npz_dir

            self.classification_task = task_data["classification_task"]
            self.classes = task_data["classes"]
            self.class_to_idx = {
                class_name: idx for idx, class_name in enumerate(self.classes)
            }

            self.data = task_data["samples"]

            if record_source is None:
                assert len(self.data) == task_data["n_samples"], (
                    f"Expected {task_data['n_samples']} samples for "
                    f"{self.classification_task}, got {len(self.data)}"
                )
            else:
                assert isinstance(record_source, list)
                assert all(isinstance(source, str) for source in record_source)

                self.data = [
                    item for item in self.data
                    if any(item[0].startswith(f"{source}__") for source in record_source)
                ]

            self.return_sample_name = return_sample_name

        def __len__(self):
            return len(self.data)

        def _get_npz_path(self, folder_name):
            npz_path = os.path.join(self.npz_dir, f"{folder_name}.npz")
            assert os.path.isfile(npz_path), f"NPZ file not found: {npz_path}"
            return npz_path

        def __getitem__(self, index):
            folder_name, target_label, years = self.data[index]
            
            with open(self._get_npz_path(folder_name), "rb") as f:
                npz_bytes = f.read()

            image = load_npz_and_process({"image": npz_bytes})["image"]

            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)

            target_idx = self.class_to_idx[target_label]
            target_idx = torch.tensor(target_idx, dtype=torch.long)

            if self.return_sample_name:
                return image, target_idx, folder_name
            else:
                return image, target_idx

    assert os.path.isfile(json_path), f"JSON file not found: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    dataloaders = {}

    for task_data in all_tasks:
        classification_task = task_data["classification_task"]

        dataset = ClassificationDataset(
            task_data=task_data,
            npz_dir=npz_dir,
            return_sample_name=return_sample_name,
            record_source=record_source,
        )

        sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=False,
        )

        dataloader.num_samples = len(dataset)
        dataloader.num_batches = len(dataloader)

        dataloaders[classification_task] = dataloader

    return dataloaders

def get_ibtracs_reg_dataset(
    json_path, npz_dir, batch_size, workers,
    return_sample_name=False,
):
    class IbtracsDataset(torch.utils.data.Dataset):
        def __init__(self, json_path, npz_dir, return_sample_name=False):
            assert os.path.isfile(json_path), f"JSON file not found: {json_path}"
            assert os.path.isdir(npz_dir), f"NPZ directory not found: {npz_dir}"
            
            self.npz_dir = npz_dir

            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            
            self.return_sample_name = return_sample_name
        
        def __len__(self):
            return len(self.data)

        def _get_npz_path(self, folder_name):
            npz_path = os.path.join(self.npz_dir, f"{folder_name}.npz")
            assert os.path.isfile(npz_path), f"NPZ file not found: {npz_path}"
            return npz_path

        def __getitem__(self, index):
            folder_name, class_name, target = self.data[index]
            
            with open(self._get_npz_path(folder_name), "rb") as f:
                npz_bytes = f.read()
            image = load_npz_and_process({"image": npz_bytes})["image"]

            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)

            target = torch.tensor(float(target), dtype=torch.float32)

            if self.return_sample_name:
                return image, class_name, target, folder_name
            else:
                return image, class_name, target

    dataset = IbtracsDataset(
        json_path=json_path,
        npz_dir=npz_dir,
        return_sample_name=return_sample_name,
    )
    sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )

    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_retrieval_val_dataset(
            args.val_data, args.val_dir_npz, args.val_dir_cap,
            caption_style="natural_event_focused", return_metadata=False,
            batch_size=args.batch_size, workers=args.workers,
            tokenizer=tokenizer,
        )

    if args.classification_val is not None:
        data["classification-val"] = get_classification_dataset(
            args.classification_val,
            args.val_dir_npz,
            args.batch_size,
            args.workers,
        )
    
    if args.ibtracs_wind_val is not None:
        data["ibtracs-wind-val"] = get_ibtracs_reg_dataset(
            args.ibtracs_wind_val,
            args.val_dir_npz,
            args.batch_size,
            args.workers,
        )
    
    if args.ibtracs_pres_val is not None:
        data["ibtracs-pres-val"] = get_ibtracs_reg_dataset(
            args.ibtracs_pres_val,
            args.val_dir_npz,
            args.batch_size,
            args.workers,
        )

    return data

import io
import numpy as np
def load_npz_and_process(sample):
    data = np.load(io.BytesIO(sample["image"]), allow_pickle=False)
    x = data["x"] # (T, V, H, W) ~ (time, variable, height, width)
    
    assert x.ndim == 4, f"Expected x to have 4 dimensions (T, V, H, W), but got {x.ndim} dimensions."
    assert x.shape[0] == 1, f"Expected T dimension to be 1, but got {x.shape[0]}."

    x_mean = ERA5_CROP_STAT["mean"] # (V,)
    x_std  = ERA5_CROP_STAT["std"]  # (V,)

    assert x_mean.ndim == 1 and x_std.ndim == 1
    assert x_mean.shape[0] == x_std.shape[0]
    assert x_mean.shape[0] == x.shape[1]
    assert np.all(x_std > 0), "x_std must be strictly positive for normalization."

    # normalize x over variable dimension
    # (T, V, H, W) - (1, V, 1, 1) / (1, V, 1, 1)
    x = (x - x_mean[None, :, None, None]) / x_std[None, :, None, None]

    if CLIMATE_INJECT_POSITION or CLIMATE_INJECT_TIME_HOUR or CLIMATE_INJECT_TIME_DOY or CLIMATE_INJECT_TIME_YEAR:
        latitude   = data["latitude"]    # (H,)
        longitude  = data["longitude"]   # (W,)
        valid_time = data["valid_time"]  # (T,)
        assert latitude.ndim == 1 and longitude.ndim == 1 and valid_time.ndim == 1
        
        T, V, H, W = x.shape

        assert latitude.shape[0]   == H
        assert longitude.shape[0]  == W
        assert valid_time.shape[0] == T

        # encode longitude with sin/cos so that it wraps properly
        longitude_rad = np.deg2rad(longitude.astype(np.float32))
        longitude_sin = np.sin(longitude_rad)
        longitude_cos = np.cos(longitude_rad)

        # encode latitude with sin/cos
        latitude_rad = np.deg2rad(latitude.astype(np.float32))
        latitude_sin = np.sin(latitude_rad)
        latitude_cos = np.cos(latitude_rad)

        # encode valid_time with cyclic day-of-year and hour-of-day
        valid_time_hour = valid_time.astype("datetime64[h]")
        valid_time_day  = valid_time.astype("datetime64[D]")
        valid_time_year = valid_time.astype("datetime64[Y]")

        hour = (valid_time_hour - valid_time_day).astype(np.int32).astype(np.float32)          # (T,)
        dayofyear = (valid_time_day - valid_time_year).astype(np.int32).astype(np.float32) + 1 # (T,)
        year = (valid_time_year.astype(np.int32) + 1970).astype(np.float32)

        hour_rad = 2.0 * np.pi * hour / 24.0
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)

        doy_rad = 2.0 * np.pi * (dayofyear - 1.0) / 365.25
        doy_sin = np.sin(doy_rad)
        doy_cos = np.cos(doy_rad)

        year = (year - 2000.0) / 25.0

        lat_sin_channel = np.broadcast_to(latitude_sin[None, None, :, None], (T, 1, H, W)) # expand latitude to (T, 1, H, W)
        lat_cos_channel = np.broadcast_to(latitude_cos[None, None, :, None], (T, 1, H, W)) # expand latitude to (T, 1, H, W)
        lon_sin_channel  = np.broadcast_to(longitude_sin[None, None, None, :], (T, 1, H, W)) # expand longitude to (T, 1, H, W)
        lon_cos_channel  = np.broadcast_to(longitude_cos[None, None, None, :], (T, 1, H, W)) # expand longitude to (T, 1, H, W)
        hour_sin_channel = np.broadcast_to(hour_sin[:, None, None, None], (T, 1, H, W)) # expand valid_time to (T, 1, H, W)
        hour_cos_channel = np.broadcast_to(hour_cos[:, None, None, None], (T, 1, H, W)) # expand valid_time to (T, 1, H, W)
        doy_sin_channel  = np.broadcast_to(doy_sin[:, None, None, None], (T, 1, H, W)) # expand valid_time to (T, 1, H, W)
        doy_cos_channel  = np.broadcast_to(doy_cos[:, None, None, None], (T, 1, H, W)) # expand valid_time to (T, 1, H, W)
        year_channel     = np.broadcast_to(year[:, None, None, None], (T, 1, H, W)) # expand valid_time to (T, 1, H, W)

        extra_channels = []
        if CLIMATE_INJECT_POSITION: # adds: lat_sin, lat_cos, lon_sin, lon_cos
            extra_channels.extend(
                [
                    lat_sin_channel,
                    lat_cos_channel,
                    lon_sin_channel,
                    lon_cos_channel,
                ]
            )
        if CLIMATE_INJECT_TIME_HOUR: # adds: hour_sin, hour_cos
            extra_channels.extend(
                [
                    hour_sin_channel,
                    hour_cos_channel,
                ]
            )
        if CLIMATE_INJECT_TIME_DOY: # adds: doy_sin, doy_cos
            extra_channels.extend(
                [
                    doy_sin_channel,
                    doy_cos_channel,
                ]
            )
        if CLIMATE_INJECT_TIME_YEAR: # adds: year
            extra_channels.extend(
                [
                    year_channel,
                ]
            )

        if extra_channels:
            x = np.concatenate([x, *extra_channels], axis=1)

    # remove time dimension, now (V, H, W)
    x = x[0]
    
    sample["image"] = torch.from_numpy(x).float()
    return sample

def load_json_and_process(tokenizer, caption_style=None, rng=None, return_raw_text=False):
    rng = rng or random.Random()

    def select_caption(sample):
        raw = sample["text"]

        if isinstance(raw, bytes):
            dict_captions = json.loads(raw.decode("utf-8"))
        elif isinstance(raw, str):
            dict_captions = json.loads(raw)
        else:
            raise TypeError(f"Unsupported type for sample['text']: {type(raw)}")
        
        if caption_style is None:
            caption = rng.choice(list(dict_captions.values()))
        else:
            if caption_style not in dict_captions:
                raise KeyError(f"{caption_style!r} not found in dict_captions")
            caption = dict_captions[caption_style]
        
        sample["text"] = tokenizer(caption)[0]
        if return_raw_text:
            sample["raw_text"] = caption
        
        return sample
    
    return select_caption

def load_npz_and_process_ret(sample):
    with np.load(io.BytesIO(sample["image"]), allow_pickle=False) as data:
        folder_name = data["folder_name"]

        if isinstance(folder_name, np.ndarray):
            if folder_name.shape == ():
                folder_name = folder_name.item()
            elif folder_name.size == 1:
                folder_name = folder_name.reshape(-1)[0]
            else:
                folder_name = folder_name.tolist()

        if isinstance(folder_name, bytes):
            folder_name = folder_name.decode("utf-8")

        folder_name = str(folder_name)

    sample = load_npz_and_process(sample)
    sample["folder_name"] = folder_name
    return sample

def load_json_and_process_ret(tokenizer, caption_style=None, rng=None):
    return load_json_and_process(tokenizer, caption_style, rng, return_raw_text=True)