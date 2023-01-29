import hashlib
import logging
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from neuro_trigger.pytorch import dataset_filters


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# type variables for type hinting
BDType = TypeVar("BDType", bound="BelleIIDataset")
T = TypeVar("T")


# TODO: should this also manage dataset versions and dataset splits -> yes splits would definitely make sense
# TODO: add preprocessing, maybe pytorch transformers
class BelleIIDataManager:
    _cache_dir = ".cache"

    def __init__(
        self, paths: List[str], out_dim: int = 2, compare_to: Optional[str] = None
    ) -> None:
        """Manages the loaded data. Can create arbitrary datasets out of the given data with given
        filters applied.


        Args:
            paths (List[str]): List of paths to the dataset
            out_dim (int, optional): Number of output neurons. Can either be 2 or 1.
                If out_dim=1 than the network is trained on z. Defaults to 2.
            compare_to (Optional[str], optional): Path to a training to which one wants to compare.
                Path should have the following format: baseline_v2/version_4. Defaults to None.
        """
        # out_dim either 2 or 1 if only z should be compared
        super().__init__()
        self.paths = paths
        self._cache_files = [f"{md5(path)}.pt" for path in self.paths]
        self.logger = logging.getLogger()
        self.out_dim = out_dim
        self.compare_to = compare_to
        self.data: Optional[Dict[str, torch.Tensor]] = None
        self.load_data(self.compare_to)

        paths_str = "\n".join(self.paths)
        self.logger.debug(f"Dataset:\n{paths_str}\nwith length {len(self)} done init")

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.data["x"])

    def load_data(self, compare_to: Optional[str] = None):
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
            "y": torch.Tensor(dt[:, 36 : 36 + self.out_dim]),
            "expert": torch.Tensor(dt[:, 6]),
            # out_dim==2 -> -4:-1:2 out_dim==1 -> -4:-3:2
            "y_hat_old": torch.Tensor(
                dt[:, -4 : (-1 if self.out_dim == 2 else -3) : 2]
            ),
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
            assert y_hat_old.shape[1] == 2
            if self.out_dim == 1:
                y_hat_old = y_hat_old[:, 0]
            self.data["y_hat_old"] = y_hat_old

    def dataset(
        self,
        filter: Optional[dataset_filters.Filter] = None,
        dataset_class: Optional[Union[partial, BDType]] = None,
    ) -> "BelleIIDataset":
        """Calculates the dataset given the configured filters and returns the corresponding dataset object.

        Args:
            filter (Optional[dataset_filters.Filter], optional): Filter that should be applied. None means that no filtering is applied. Defaults to None.
            dataset_class (Optional[Union[partial, BDType]], optional): Dataset class that should be used. Defaults to None which means that `BelleIIDataset` is used.

        Returns:
            BelleIIDataset: The resulting filtered dataset object
        """
        # default values
        self.logger.debug(f"Size before filter: {len(self)}")
        filter = filter or dataset_filters.IdentityFilter()
        dataset_class = dataset_class or BelleIIDataset

        keep = torch.where(filter.fltr(self.data))

        data = {key: val[keep] for key, val in self.data.items()}
        self.logger.debug(f"Size after filter: {len(data['x'])}")
        return dataset_class(data)

    def expert_dataset(
        self,
        expert: int = -1,
        filter: Optional[dataset_filters.Filter] = None,
        dataset_class: Optional[Union[partial, BDType]] = None,
    ) -> "BelleIIDataset":
        """Similar to the `dataset` function except that the ExpertFilter is automatically applied with the given expert.

        Args:
            expert (int, optional): Expert number, -1 means no expert filtering (one big dataset). Defaults to -1.
            filter (Optional[dataset_filters.Filter], optional): Additional filters to apply, None means no other filters should be applied. Defaults to None.
            dataset_class (Optional[Union[partial, BDType]], optional): Dataset class that should be used. Defaults to None which means that `BelleIIDataset` is used.

        Returns:
            BelleIIDataset: The resulting filtered dataset object
        """
        filter = filter or dataset_filters.IdentityFilter()
        filter = dataset_filters.ConCatFilter(
            [filter, dataset_filters.ExpertFilter(expert=expert)]
        )
        dataset = self.dataset(filter, dataset_class)
        self.logger.debug(f"Expert #{expert} with length {len(dataset)} created")
        return dataset

    def get_data_array(self) -> torch.Tensor:
        """Returns the data tensor of the given dataset.

        This function uses a caching mechanism to speed up data loading.
        If a dataset is open the first time a cache object is created which is identified by the file's md5 hash and which can be open much faster.
        Thus, the next time this function is called it is detected that the dataset has been cached and can be opened much faster.

        """
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
        """This function is used whenever a dataset has not yet been cached and needs to be read from a CSV file."""
        return torch.Tensor(np.loadtxt(path, skiprows=2))

    def save_cache(self, dt: Union[torch.Tensor, np.array], cache_file: str):
        """Saves a dataset tensor with torch.save to be able to read it in much faster than parsing a CSV file.

        Args:
            dt (Union[torch.Tensor, np.array]): dataset tensor to save
            cache_file (str): file path of the CSV file that should be saved, in order to calculate the md5 hash of it which is used for cache file's name.
        """
        if not Path(self._cache_dir).exists():
            Path(self._cache_dir).mkdir()
        with open(os.path.join(self._cache_dir, cache_file), "wb") as f:
            torch.save(dt, f)

    def open_cache(self, cache_file: str) -> torch.Tensor:
        """Opens a cached dataset given its file name. This file name must be inferred with the md5 hash with the original dataset's filename"""
        with open(os.path.join(self._cache_dir, cache_file), "rb") as f:
            dt = torch.load(f)
            if isinstance(dt, np.ndarray):
                return torch.Tensor(dt)
            return dt


class BelleIIDataset(Dataset):
    """Dataset representation of the Belle II CSV dataset"""

    Z_SCALING = [-100, 100]
    THETA_SCALING = [10, 170]

    def __init__(self, data: Dict[str, torch.Tensor]):
        """
        Args:
            data (Dict[str, torch.Tensor]): data in dictionary form as returned by `load_data` in BelleIIDataManager
        """
        self.data = data

    def __len__(self) -> int:
        """Returns dataset length"""
        return len(self.data["x"])

    def __getitem__(self, idx: int) -> Tuple[float, float, float, int]:
        """Returns the ith sample

        Args:
            idx (int): index

        Raises:
            IndexError: For idx < 0 or idx > len

        Returns:
            Tuple[float, float, float, int]: x, y, y_hat_old, idx
        """
        if idx >= len(self) or idx < 0:
            raise IndexError()
        return (
            self.data["x"][idx],
            self.data["y"][idx],
            self.data["y_hat_old"][idx],
            self.data["idx"][idx],
        )

    @staticmethod
    def scale(
        x: Union[float, torch.Tensor],
        lower: float,
        upper: float,
        lower_new: float,
        upper_new: float,
    ) -> Union[torch.Tensor, float]:
        """Scales input linearly: [lower, upper] -> [lower_new, upper_new]

        Args:
            x (Union[float, torch.Tensor]): Number or tensor that should be scaled
            lower (float): lower bound of the current interval of x
            upper (float): upper bound of the current interval of x
            lower_new (float): lower bound of the new interval where x should be mapped to
            upper_new (float): upper bound of the new interval where x should be mapped to

        Returns:
            Union[torch.Tensor, float]: mapped x
        """
        # linear scaling
        # first scale to [0, 1], then scale to new interval
        return ((x - lower) / (upper - lower)) * (upper_new - lower_new) + lower_new

    @staticmethod
    def to_physics(x: torch.Tensor) -> torch.Tensor:
        """Mapps x from the [-1, 1] interval to the real physical interval in the measured units from Belle II

        If x has only one dimension it is assumed that this dimension represends z. Thus a mapping from
        [-1, 1] to [-100, 100] (centimenters) is performed.

        If x has two dimensions, it is assumed that the first dimension is z and the second is theta.
        z is mapped as described above. Theta is mapped as follows: [-1, 1] -> [10, 170] (degree)

        Args:
            x (torch.Tensor): z (theta) tensor that should be mapped to the physical measured interval

        Returns:
            torch.Tensor: Mapped z (and theta) to the physical interval in the respective units
        """
        x_ = x.clone()
        x_[:, 0] = BelleIIDataset.scale(x_[:, 0], -1, 1, *BelleIIDataset.Z_SCALING)
        if x_.shape[1] > 1:
            x_[:, 1] = BelleIIDataset.scale(
                x_[:, 1], -1, 1, *BelleIIDataset.THETA_SCALING
            )
        return x_

    @staticmethod
    def from_physics(x: torch.Tensor) -> torch.Tensor:
        """Like to_physics but in the opposit direction: Map values to [-1, 1] interval"""
        x_ = x.clone()
        x_[:, 0] = BelleIIDataset.scale(x_[:, 0], *BelleIIDataset.Z_SCALING, -1, 1)
        if x_.shape[1] > 1:
            x_[:, 1] = BelleIIDataset.scale(
                x_[:, 1], *BelleIIDataset.THETA_SCALING, -1, 1
            )
        return x_

    @property
    def requires_shuffle(self) -> bool:
        """Defines whether the dataset needs to be shuffled by the dataloader or is inherently
        shuffeled as the samples are randomly picked anyway and thus dataloader shuffling is not required.
        """
        return True


class BelleIIDistDataset(BelleIIDataset):
    """Implements dataset reweighting with a given scipy distribution."""

    # TODO: should this return batches?

    def __init__(
        self, *args, dist, n_buckets: int = 21, inf_bounds: bool = False, **kwargs
    ) -> None:
        """
        Args:
            dist (scipy.stats): Scipy distribution to sample the buckets
            n_buckets (int, optional): Number of buckets for sample picking. Defaults to 21.
            inf_bounds (bool, optional): Wether the most left and most right bucket include the probability
                towards infinity or whether they dont include this probability and normalize the resulting distribution
                to a sum of 1. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.sort_z: List[Tuple[int, float]] = [
            (idx, i[0].item()) for idx, i in enumerate(self.data["y"])
        ]
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

        self.probs = [
            self.get_prob_for_bounds(*self.get_bounds(bucket))
            for bucket in self.bucket_idx
        ]
        if not self.inf_bounds:
            # normalize to one
            self.probs = [i / sum(self.probs) for i in self.probs]

    def get_prob_for_bounds(self, lower: float, upper: float) -> float:
        """Given the scipy distribution, this calulates the CDF difference between lower and upper
        bound

        _extended_summary_

        Args:
            lower (float): lower bound the wanted interval of the density function
            upper (float): uppper bound the wanted interval of the density function

        Returns:
            float: Integrated probability
        """
        return self.dist.cdf(upper) - self.dist.cdf(lower)

    def get_bounds(
        self, bucket: int, inf_bounds: Optional[bool] = None
    ) -> Tuple[float, float]:
        """Interval bounds of the given bucket distributed within [-1,1].

        Args:
            bucket (int): Bucket number
            inf_bounds (Optional[bool], optional): If set to False the outer most buckets are
                unbounded towards +/- inf.
                Example: bucket=0,inf_bounds=False,self.n_buckets=2 -> [-inf, 0]
                If set to True the interval is closed at -/+ 1.
                Example: bucket=0,inf_bounds=True,self.n_buckets=2 -> [-1, 0]
                Defaults to None which takes the
                value of `self.inf_bounds`.

        Returns:
            Tuple[float, float]: The interval which the bucket represents.
        """
        if inf_bounds is None:
            inf_bounds = self.inf_bounds
        lower = 2 * (bucket / self.n_buckets - 0.5)
        upper = lower + 2 / self.n_buckets
        if inf_bounds:
            if math.isclose(lower, -1):
                lower = -math.inf
            if math.isclose(upper, 1):
                upper = math.inf
        return lower, upper

    def get_bucket(self, z: float) -> int:
        """Returns the bucket for a given z."""
        if z == 1:
            # last bucket should include the 1.0: [x, 1.0]
            return self.n_buckets - 1
        return math.floor((z / 2 + 0.5) * self.n_buckets)

    def __len__(self) -> int:
        return len(self.data["x"])

    @property
    def requires_shuffle(self) -> bool:
        """Does not require further shuffeling by the dataloader as samples are randomly picked anyway."""
        return False

    def __getitem__(self, idx: int) -> Tuple[float, float, float, float]:
        """Returns a radomly picked sample according to the following strategy:

        1. Sample a z-bucket according to the given scipy distribution e.g. normal or uniform distribution.
        2. Within the bucket the a sample is uniforml randomly selected.

        Args:
            idx (int): ith sample, ignored

        Raises:
            IndexError: Used to decide the end of an epoch: Not when the every sample has been seen
                but rather when we just iterated n times.

        Returns:
            Tuple[float, float, float, float]: Data sample: x, y, y_hat_old, idx
        """
        if idx >= len(self):
            raise IndexError()
        # sample a bucket
        bucket = np.random.choice(self.bucket_idx, p=self.probs)

        # sample uniformly from that bucket
        b = self.buckets[bucket]
        idx = self.uniform_random_choice(b)
        return (
            self.data["x"][idx],
            self.data["y"][idx],
            self.data["y_hat_old"][idx],
            self.data["idx"][idx],
        )

    def uniform_random_choice(self, a: Iterable[T]) -> T:
        """Selects uniformly random sample from array/Iterable a.
        (T is a type variable)

        Note that somehow np.random.choice scales very badly for large arrays, so we rather write
        the two lines ourself!

        Args:
            a (Iterable[T]): Array/Iterable where one wants to get a random sample

        Returns:
            T: Selected sample.
        """
        idx = random.randint(0, len(a) - 1)
        return a[idx]
