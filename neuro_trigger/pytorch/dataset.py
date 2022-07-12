from functools import partial
import math
import random
from typing import Iterable, Optional, Tuple, TypeVar, Union, List, Dict
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


# class DatasetPath:
#     SPLIT = ("train", "val", "test")
#     # we only work with static splits
#     def __init__(self, base_path: str, file_name_f_str: str = "neuroresults_random{}.gz", split: Optional[List[int]] = None) -> None:
#         self.base_path = base_path
#         self.file_name = file_name_f_str
#         # default split index is 1, 2, 3
#         self.split = split or [1, 2, 3]

#     def get_path_by_index(self, index):
#         return os.path.join(self.base_path, self.file_name.format(self.split[index]))


#     def __getitem__(self, ref):
#         if isinstance(ref, str) and ref in self.SPLIT:
#             index = self.SPLIT.index(ref)
#             return self.get_path_by_index(index)
#         elif isinstance(ref, int) and ref >= 0 and ref < len(self.split):
#             return self.get_path_by_index(ref)
#         else:
#             raise RuntimeError("Invalid split reference.")

#     def __str__(self) -> str:
#         "\n".join([self.get_path_by_index(i) for i in self.split])

#     # TODO: think if such a method makes sense
#     @staticmethod
#     def create_static_random_split():
#         pass

# type variables for type hinting
BDType = TypeVar('BDType', bound="BelleIIDataset")
T = TypeVar('T')


# TODO: support for merging several files
# TODO: should this also manage dataset versions and dataset splits -> yes splits would definatly make sense
# TODO: add preprocessing, maybe pytorch transformers
class BelleIIDataManager:
    _cache_dir = ".cache"
    def __init__(self, paths: List[str], logger: logging.Logger, out_dim: int = 2, compare_to: Optional[str] = None) -> None:
        """Manages the loaded data. Can create arbitrary datasets out of the given data with given
        filters applied.


        Args:
            paths (List[str]): List of paths to the dataset
            logger (logging.Logger): python logger for debug output
            out_dim (int, optional): Numer of output neurons. Can either be 2 or 1.
                If out_dim=1 than the network is trained on z. Defaults to 2.
            compare_to (Optional[str], optional): Path to a training to which one wants to compare.
                Path should have the following format: baseline_v2/version_4. Defaults to None.
        """
        # out_dim either 2 or 1 if only z should be compared
        super().__init__()
        self.paths = paths
        self._cache_files = [f"{md5(path)}.pt" for path in self.paths]
        self.logger = logger
        self.out_dim = out_dim
        self.compare_to = compare_to 
        self.load_data(self.compare_to)
        self.data: Optional[Dict[str, torch.Tensor]] = None

        paths_str = '\n'.join(self.paths)
        self.logger.debug(
            f"Dataset:\n{paths_str}\nwith length {len(self)} done init")

    def __len__(self) -> int:
        """Length of the dataset
        """
        return len(self.data["x"])


    def load_data(self, compare_to: Optional[str]=None):
        """Loads the csv data into self.data.

        Args:
            compare_to (Optional[str], optional): Same as in __init__. Defaults to None.
        """
        # open and concatenate datasets
        dt = self.get_data_array()

        # create easy accessable dictionary
        self.data = {
            "x": torch.Tensor(dt[:, 9:36]),
            # only 36:37 if only z (out_dim=2)
            "y": torch.Tensor(dt[:, 36:36+self.out_dim]),
            "expert": torch.Tensor(dt[:, 6]),
            # out_dim==2 -> -4:-1:2 out_dim==1 -> -4:-3:2
            "y_hat_old": torch.Tensor(dt[:, -4:(-1 if self.out_dim == 2 else -3):2]),
            # TODO: problem this does not work for numbers larger than 2^24+1 (16 777 217)
            # as the float can represent ints any more
            "idx": torch.arange(dt.shape[0]),
            "event": torch.Tensor(dt[:, 3]),
            "track": torch.Tensor(dt[:, 4]),
            "ntracks": torch.Tensor(dt[:, 5]),
        }

        # add prediction outputs from old networks
        if compare_to:
            # when we want to compare to different predictions
            with open(compare_to, "rb") as f:
                # filter for the correct indicies
                y_hat_old = torch.load(f)[self.data["idx"]]
            # check the dimension correctness
            assert(y_hat_old.shape[1] == 2)
            if self.out_dim == 1:
                y_hat_old = y_hat_old[:,0]
            self.data["y_hat_old"] = y_hat_old

    def dataset(self, filter: Optional[dataset_filters.Filter] = None, dataset_class: Optional[Union[partial, BDType]] = None) -> "BelleIIDataset":
        # default values
        self.logger.debug(f"Size before filter: {len(self)}")
        filter = filter or dataset_filters.IdentityFilter()
        dataset_class = dataset_class or BelleIIDataset

        keep = torch.where(filter.fltr(self.data))

        data = {key: val[keep] for key, val in self.data.items()}
        self.logger.debug(f"Size after filter: {len(data['x'])}")
        return dataset_class(data)

    def expert_dataset(self, expert: int = -1, filter: Optional[dataset_filters.Filter] = None, dataset_class: Optional[Union[partial, BDType]] = None) -> "BelleIIDataset":
        filter = filter or dataset_filters.IdentityFilter()
        filter = dataset_filters.ConCatFilter([filter, dataset_filters.ExpertFilter(expert=expert)])
        dataset = self.dataset(filter, dataset_class)
        self.logger.debug(
            f"Expert #{expert} with length {len(dataset)} created")
        return dataset

    def get_data_array(self) -> torch.Tensor:
        dts = []
        for cache_file, path in zip(self._cache_files, self.paths):
            # cache mechanism
            if Path(os.path.join(self._cache_dir, cache_file)).exists():
                self.logger.debug(f"{path} already cached, loading it.")
                dts.append(self.open_cache(cache_file))
            else:
                dt = self.read_from_csv(path)
                self.save_cache(dt, cache_file)
                dts.append(dt)
        return torch.cat(dts, dim=0)

    def read_from_csv(self, path: str) -> torch.Tensor:
        return torch.Tensor(np.loadtxt(path, skiprows=2))


    def save_cache(self, dt: Union[torch.Tensor, np.array], cache_file: str):
        if not Path(self._cache_dir).exists():
            Path(self._cache_dir).mkdir()
        with open(os.path.join(self._cache_dir, cache_file), "wb") as f:
            torch.save(dt, f)

    def open_cache(self, cache_file: str) -> torch.Tensor:
        with open(os.path.join(self._cache_dir, cache_file), "rb") as f:
            dt = torch.load(f)
            if isinstance(dt, np.ndarray):
                return torch.Tensor(dt)
            return dt


class BelleIIDataset(Dataset):
    Z_SCALING = [-100, 100]
    THETA_SCALING = [10, 170]

    def __init__(self, data: Dict[str, torch.Tensor]):
        self. data = data

    def __len__(self) -> int:
        return len(self.data["x"])

    def __getitem__(self, idx: int) -> Tuple[float, float, float, int]:
        if idx >= len(self):
            raise IndexError()
        return self.data["x"][idx], self.data["y"][idx], self.data["y_hat_old"][idx], self.data["idx"][idx]

    @staticmethod
    def scale(x: Union[float, torch.Tensor], lower: float, upper: float, lower_new: float, upper_new: float) -> Union[torch.Tensor, float]:
        # linear scaling
        # first scale to [0, 1], then scale to new interval
        return ((x-lower) / (upper-lower)) * (upper_new-lower_new) + lower_new

    @staticmethod
    def to_physics(x: torch.Tensor) -> torch.Tensor:
        x_ = x.clone()
        x_[:,0] = BelleIIDataset.scale(x_[:,0], -1, 1, *BelleIIDataset.Z_SCALING)
        if x_.shape[1] > 1:
            x_[:,1] = BelleIIDataset.scale(x_[:,1], -1, 1, *BelleIIDataset.THETA_SCALING)
        return x_

    @staticmethod
    def from_physics(x: torch.Tensor) -> torch.Tensor:
        x_ = x.clone()
        x_[:,0] = BelleIIDataset.scale(x_[:,0], *BelleIIDataset.Z_SCALING, -1, 1)
        if x_.shape[1] > 1:
            x_[:,1] = BelleIIDataset.scale(x_[:,1], *BelleIIDataset.THETA_SCALING, -1, 1)
        return x_

    @property
    def requires_shuffle(self) -> bool:
        return True

class BelleIIDistDataset(BelleIIDataset):
    # TODO: should this return batches?

    def __init__(self, *args, dist, n_buckets: int = 21, inf_bounds: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sort_z: List[Tuple[int, float]] = [(idx, i[0].item()) for idx, i in enumerate(self.data["y"])]
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


    def get_prob_for_bounds(self, lower: float, upper: float) -> float:
        return self.dist.cdf(upper) - self.dist.cdf(lower)
    

    def get_bounds(self, bucket: int, inf_bounds: Optional[bool]=None) -> Tuple[float, float]:
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


    def get_bucket(self, z: float) -> int:
        return math.floor((z/2 + 0.5)*self.n_buckets)

    def __len__(self) -> int:
        return len(self.data["x"])

    @property
    def requires_shuffle(self) -> bool:
        # does not require further shuffeling by the dataloader
        return False

    def __getitem__(self, idx: int) -> Tuple[float, float, float, float]:
        if idx >= len(self):
            raise IndexError()
        # sample a bucket
        bucket = np.random.choice(self.bucket_idx, p=self.probs)

        # sample uniformly from that bucket
        b = self.buckets[bucket]
        idx = self.uniform_random_choice(b)
        return self.data["x"][idx], self.data["y"][idx], self.data["y_hat_old"][idx], self.data["idx"][idx]

    def uniform_random_choice(self, a: Iterable[T]) -> T:
        # Note somehow np.random.choice scales very badly for large arrays
        # so we rather do the two lines our self
        idx = random.randint(0, len(a)-1)
        return a[idx]

