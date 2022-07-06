import math
import random
from typing import Optional, Tuple, Union, List
from torch.utils.data import Dataset
from pathlib import Path
import torch
from pathlib import Path
import os
import logging
import numpy as np
from neuro_trigger.pytorch import dataset_filters

import hashlib

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class DatasetPath:
    SPLIT = ("train", "val", "test")
    # we only work with static splits
    def __init__(self, base_path: str, file_name_f_str: str = "neuroresults_random{}.gz", split: Optional[List[int]] = None) -> None:
        self.base_path = base_path
        self.file_name = file_name_f_str
        # default split index is 1, 2, 3
        self.split = split or [1, 2, 3]

    def get_path_by_index(self, index):
        return os.path.join(self.base_path, self.file_name.format(self.split[index]))


    def __getitem__(self, ref):
        if isinstance(ref, str) and ref in self.SPLIT:
            index = self.SPLIT.index(ref)
            return self.get_path_by_index(index)
        elif isinstance(ref, int) and ref >= 0 and ref < len(self.split):
            return self.get_path_by_index(ref)
        else:
            raise RuntimeError("Invalid split reference.")

    def __str__(self) -> str:
        "\n".join([self.get_path_by_index(i) for i in self.split])

    # TODO: think if such a method makes sense
    @staticmethod
    def create_static_random_split():
        pass



# TODO: support for merging several files
# TODO: should this also manage dataset versions and dataset splits -> yes splits would definatly make sense
# TODO: add preprocessing, maybe pytorch transformers
class BelleIIDataManager:
    _cache_dir = ".cache"
    def __init__(self, path: str, logger: logging.Logger, out_dim: int = 2, compare_to: Optional[str] = None) -> None:
        # filter must be a function that receives a dictionary of the form created by the init_data function
        # it should return a filtered variant of this dataset in the same dictionary form
        # out_dim either 2 or 1 if only z should be compared
        super().__init__()
        self.path = path
        print(path)
        self._cache_file = f"{md5(self.path)}.pt"
        self.logger = logger
        self.out_dim = out_dim
        self.compare_to = compare_to 
        self.load_data(self.compare_to)

        self.logger.debug(
            f"Dataset {self.path} with length {len(self)} done init")
        print("done")

    def __len__(self):
        return len(self.data["x"])


    def load_data(self, compare_to=None):
        # cache mechanism
        if Path(os.path.join(self._cache_dir, self._cache_file)).exists():
            self.logger.debug("Already cached, loading it")
            dt = self.open()
        else:
            dt = self.get_data_array()
            self.save(dt)

        self.data = {
            "x": torch.Tensor(dt[:, 9:36]),
            # only 36:37 if only z (out_dim=2)
            "y": torch.Tensor(dt[:, 36:36+self.out_dim]),
            "expert": torch.Tensor(dt[:, 6]),
            # out_dim==2 -> -4:-1:2 out_dim==1 -> -4:-3:2
            "y_hat_old": torch.Tensor(dt[:, -4:(-1 if self.out_dim == 2 else -3):2]),
            "idx": torch.arange(dt.shape[0]),
            "event": torch.Tensor(dt[:, 3]),
            "track": torch.Tensor(dt[:, 4]),
            "ntracks": torch.Tensor(dt[:, 5]),
        }

        if compare_to:
            # when we want to compare to different predictions
            with open(compare_to, "rb") as f:
                # filter for the correct indicies
                y_hat_old = torch.load(f)[self.data["idx"]]
            if self.out_dim == 1:
                y_hat_old = y_hat_old[:,0]
            self.data["y_hat_old"] = y_hat_old

    def dataset(self, filter=None, dataset_class=None):
        # default values
        self.logger.debug(f"Size before filter: {len(self)}")
        filter = filter or dataset_filters.IdenityFilter()
        dataset_class = dataset_class or BelleIIDataset

        keep = torch.where(filter.fltr(self.data))

        data = {key: val[keep] for key, val in self.data.items()}
        self.logger.debug(f"Size after filter: {len(data['x'])}")
        return dataset_class(data)

    def expert_dataset(self, expert=-1, filter=None, dataset_class=None):
        filter = filter or dataset_filters.IdenityFilter()
        filter = dataset_filters.ConCatFilter([filter, dataset_filters.ExpertFilter(expert=expert)])
        dataset = self.dataset(filter, dataset_class)
        self.logger.debug(
            f"Expert #{expert} with length {len(dataset)} created")
        return dataset


    def get_data_array(self):
        # also used in utils
        dt = np.loadtxt(self.path, skiprows=2)
        return dt


    def save(self, dt):
        if not Path(self._cache_dir).exists():
            Path(self._cache_dir).mkdir()
        with open(os.path.join(self._cache_dir, self._cache_file), "wb") as f:
            torch.save(dt, f)

    def open(self):
        with open(os.path.join(self._cache_dir, self._cache_file), "rb") as f:
            return torch.load(f)


class BelleIIDataset(Dataset):
    Z_SCALING = [-100, 100]
    THETA_SCALING = [10, 170]

    def __init__(self, data):
        self. data = data

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError()
        return self.data["x"][idx], self.data["y"][idx], self.data["y_hat_old"][idx], self.data["idx"][idx]

    @staticmethod
    def scale(x: Union[float, torch.Tensor], lower: float, upper: float, lower_new: float, upper_new: float):
        # linear scaling
        # first scale to [0, 1], then scale to new interval
        return ((x-lower) / (upper-lower)) * (upper_new-lower_new) + lower_new

    @staticmethod
    def to_physics(x: torch.Tensor):
        x_ = x.clone()
        x_[:,0] = BelleIIDataset.scale(x_[:,0], -1, 1, *BelleIIDataset.Z_SCALING)
        if x_.shape[1] > 1:
            x_[:,1] = BelleIIDataset.scale(x_[:,1], -1, 1, *BelleIIDataset.THETA_SCALING)
        return x_

    @staticmethod
    def from_physics(x: torch.Tensor):
        x_ = x.clone()
        x_[:,0] = BelleIIDataset.scale(x_[:,0], *BelleIIDataset.Z_SCALING, -1, 1)
        if x_.shape[1] > 1:
            x_[:,1] = BelleIIDataset.scale(x_[:,1], *BelleIIDataset.THETA_SCALING, -1, 1)
        return x_

    @property
    def requires_shuffle(self):
        return True

class BelleIIDistDataset(BelleIIDataset):
    # TODO: should this return batches?

    def __init__(self, *args, dist, n_buckets=21, inf_bounds=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sort_z = [(idx, i[0].item()) for idx, i in enumerate(self.data["y"])]
        self.sort_z = sorted(self.sort_z, key=lambda x: x[1])
        self.n_buckets = n_buckets
        self.buckets = {}
        for idx, z in self.sort_z:
            bucket = self.get_bucket(z)
            if bucket not in self.buckets:
                self.buckets[self.get_bucket(z)] = []

            self.buckets[self.get_bucket(z)].append(idx)

        self.bucket_idx = np.arange(len(self.buckets))
        self.dist = dist
        self.inf_bounds = inf_bounds

        self.probs = [self.get_prob_for_bounds(*self.get_bounds(bucket)) for bucket in self.bucket_idx]
        if not self.inf_bounds:
            # normalize to one
            self.probs = [i/sum(self.probs) for i in self.probs]


    def get_prob_for_bounds(self, lower: float, upper: float):
        return self.dist.cdf(upper) - self.dist.cdf(lower)
    

    def get_bounds(self, bucket: int, inf_bounds: Optional[bool]=None):
        if inf_bounds is None:
            inf_bounds = self.inf_bounds
        lower = 2*(bucket/self.n_buckets - 0.5)
        upper = lower + 2/self.n_buckets
        if inf_bounds:
        if math.isclose(lower, -1):
            lower = -math.inf
        if math.isclose(upper, 1):
            upper = math.inf
        return lower, upper


    def get_bucket(self, z: float):
        return math.floor((z/2 + 0.5)*self.n_buckets)

    def __len__(self):
        return len(self.data["x"])

    @property
    def requires_shuffle(self):
        # does not require further shuffeling by the dataloader
        return False

    def __getitem__(self, idx: int) -> Tuple[float, ...]:
        if idx >= len(self):
            raise IndexError()
        # sample a bucket
        bucket = np.random.choice(self.bucket_idx, p=self.probs)

        # sample uniformly from that bucket
        b = self.buckets[bucket]
        idx = self.uniform_random_choice(b)
        return self.data["x"][idx], self.data["y"][idx], self.data["y_hat_old"][idx], self.data["idx"][idx]

    def uniform_random_choice(self, a: List):
        # Note somehow np.random.choice scales very badly for large arrays
        # so we rather do the two lines our self
        idx = random.randint(0, len(a)-1)
        return a[idx]

