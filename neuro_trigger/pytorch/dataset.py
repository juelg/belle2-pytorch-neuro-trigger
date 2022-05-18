import math
from typing import Optional, Union
from torch.utils.data import Dataset
from pathlib import Path
import torch
import linecache
from multiprocessing import Pool
import time
from pathlib import Path
import os
import logging
import gzip
import numpy as np

import hashlib

class filters
    # filter function
    @staticmethod
    def filter_max_2_events(data):
        b_idx = data["ntracks"] > 2
        for key in data:
            data[key] = data[key][~b_idx]
        return data

    @staticmethod
    def filter_duplicate_events(data):
        # create map event,y -> tracks
        event_map = {}
        for idx, (e, y) in enumerate(zip(data["event"], data["y"])):
            ey = f"{e},{y[0]},{y[1]}"
            if ey not in event_map:
                event_map[ey] = []
            event_map[ey].append(idx)
        keep_idx = []
        for key, value in event_map.items():
            keep_idx.append(value[0])
        keep_idx = np.array(keep_idx)

        for key in data:
            data[key] = data[key][keep_idx]

        return dataa

    @staticmethod
    def no_filter(data):
        return data


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()



# TODO: add preprocessing
class BelleIIBetter(Dataset):
    _cache_dir = ".cache"
    Z_SCALING = [-100, 100]
    THETA_SCALING = [10, 170]
    def __init__(self, path: str, logger: logging.Logger, out_dim: int = 2, compare_to: Optional[str] = None, fltr=None) -> None:
        # filter must be a function that receives a dictionary of the form created by the init_data function
        # it should return a filtered variant of this dataset in the same dictionary form
        # out_dim either 2 or 1 if only z should be compared
        super().__init__()
        self.path = path
        print(path)
        self._cache_file = f"{md5(self.path)}.pt"
        self.logger = logger
        self.out_dim = out_dim
        self.filter = fltr or (lambda x: x)
        self.compare_to = compare_to 
        self.init_data(self.filter, self.compare_to)



        self.logger.debug(
            f"Dataset {self.path} with length {len(self)} done init")
        print("done")

    def init_data(self, filter=None, compare_to=None):
        filter = filter or (lambda x: x)
        if Path(os.path.join(self._cache_dir, self._cache_file)).exists():
            self.logger.debug("Already cached, loading it")
            self.data = filter(self.open())
        else:
        # if True:
            dt = self.get_data_array()
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
            self.save()
            self.data = filter(self.data)


        if compare_to:
            # when we want to compare to different predictions
            with open(compare_to, "rb") as f:
                # filter for the correct indicies
                y_hat_old = torch.load(f)[self.data["idx"]]
            if self.out_dim == 1:
                y_hat_old = y_hat_old[:,0]
            self.data["y_hat_old"] = y_hat_old


    @property
    def requires_shuffle(self):
        return True

    def get_data_array(self):
        # also used in utils
        dt = np.loadtxt(self.path, skiprows=2)
        return dt


    def save(self):
        if not Path(self._cache_dir).exists():
            Path(self._cache_dir).mkdir()
        with open(os.path.join(self._cache_dir, self._cache_file), "wb") as f:
            torch.save(self.data, f)

    def open(self):
        with open(os.path.join(self._cache_dir, self._cache_file), "rb") as f:
            return torch.load(f)


    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx: int):
        return self.data["x"][idx], self.data["y"][idx], self.data["y_hat_old"][idx], self.data["idx"][idx]

    @staticmethod
    def scale(x: Union[float, torch.Tensor], lower: float, upper: float, lower_new: float, upper_new: float):
        # linear scaling
        # first scale to [0, 1], then scale to new interval
        return ((x-lower) / (upper-lower)) * (upper_new-lower_new) + lower_new

    @staticmethod
    def to_physics(x: torch.Tensor):
        x_ = x.clone()
        x_[:,0] = BelleIIBetter.scale(x_[:,0], -1, 1, *BelleIIBetter.Z_SCALING)
        if x_.shape[1] > 1:
            x_[:,1] = BelleIIBetter.scale(x_[:,1], -1, 1, *BelleIIBetter.THETA_SCALING)
        return x_

    @staticmethod
    def from_physics(x: torch.Tensor):
        x_ = x.clone()
        x_[:,0] = BelleIIBetter.scale(x_[:,0], *BelleIIBetter.Z_SCALING, -1, 1)
        if x_.shape[1] > 1:
            x_[:,1] = BelleIIBetter.scale(x_[:,1], *BelleIIBetter.THETA_SCALING, -1, 1)
        return x_


class BelleIIBetterExpert(BelleIIBetter):
    def __init__(self, expert: int, *args, **kwargs) -> None:
        self.expert = expert
        super().__init__(*args, **kwargs)

        # # filter out all samples that do not belong to this expert
        # # create index map
        # keep = torch.where(self.data["expert"] == self.expert)
        # # overwrite in order to get back memory from unused data
        # self.data = {key: val[keep] for key, val in self.data.items()}
        # # senity check
        # assert (self.data["expert"] == self.expert).all()

        self.logger.debug(
            f"Dataset {self.path} expert #{self.expert} with length {len(self)} done init")

    def init_data(self, filter=None, compare_to=None):
        super().init_data(filter, compare_to)

        # filter out all samples that do not belong to this expert
        # create index map
        keep = torch.where(self.data["expert"] == self.expert)
        # overwrite in order to get back memory from unused data
        self.data = {key: val[keep] for key, val in self.data.items()}
        # senity check
        assert (self.data["expert"] == self.expert).all()

class BelleIIBetterExpertDist(BelleIIBetterExpert):
    # TODO: should this return batches?

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(dist, *args, n_buckets=21, **kwargs)
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

        self.probs = [self.get_prob_for_bounds(*self.get_bounds(bucket)) for bucket in self.bucket_idx]


    def get_prob_for_bounds(self, lower, upper):
        return self.dist.cdf(upper) - self.dist.cdf(lower)
    

    def get_bounds(self, bucket):
        lower = 2*(bucket/self.n_buckets - 0.5)
        upper = lower + 2/self.n_buckets
        if math.isclose(lower, -1):
            lower = -math.inf
        if math.isclose(upper, 1):
            upper = math.inf
        return lower, upper

    def get_bucket(self, z):
        return math.floor((z/2 + 0.5)*self.n_buckets)

    def __len__(self):
        return len(self.data["x"])

    @property
    def requires_shuffle(self):
        # does not require further shuffeling by the dataloader
        return False

    def __getitem__(self, _):
        # sample a bucket
        # print(self.probs)
        # print(sum(self.probs))
        bucket = np.random.choice(self.bucket_idx, p=self.probs)

        # sample uniformly from that bucket
        idx = np.random.choice(self.buckets[bucket])
        return self.data["x"][idx], self.data["y"][idx], self.data["y_hat_old"][idx], self.data["idx"][idx]



if __name__ == "__main__":
    b = BelleIIBetter(
        "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random1.gz", logging.getLogger("test"))
    for i in b[:1]:
        print(i)
