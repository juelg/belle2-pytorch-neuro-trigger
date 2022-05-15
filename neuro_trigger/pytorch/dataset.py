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


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class BelleII(Dataset):
    # DEPRECATED since 0.3
    _cache_dir = ".cache"
    # _cache_file = "data.pt"
    def __init__(self, path, logger, in_ram=True) -> None:
        super().__init__()
        # with gzip.open(path) as f:
        #     self.l = sum(1 for _ in f)
        self.path = path
        self.in_ram = in_ram
        self._cache_file = f"{md5(self.path)}.pt"
        self.logger = logger
        # self.x, self.y, self.expert, self.meta = (None for _ in range(4))
        self.data = {}
        if self.in_ram:
            if Path(os.path.join(self._cache_dir, self._cache_file)).exists():
                self.logger.debug("Already cached, loading it")
                self.data = self.open()
            else:
                self.logger.debug(
                    f"{self.path} not cached yet, start caching, this might take a while")
                t1 = time.time()
                with Pool(10) as p:
                    with gzip.open(self.path) as f:
                        splitted = p.map(self.line2data, f.readlines())
                t2 = time.time()
                self.logger.debug("Time for caching: %f.1", t2-t1)

                # self.x = torch.Tensor([i[0] for i in splitted])
                # self.y = torch.Tensor([i[1] for i in splitted])
                # self.expert = torch.Tensor([i[2] for i in splitted])
                # self.meta = torch.Tensor([i[3] for i in splitted])
                self.data = {
                    "x": torch.Tensor([i[0] for i in splitted]),
                    "y": torch.Tensor([i[1] for i in splitted]),
                    "expert": torch.Tensor([i[2] for i in splitted]),
                    # "meta": torch.Tensor([i[3] for i in splitted]),
                }
                self.save()
                self.logger.debug("Done saving the cache blob")
        self.logger.debug(f"Dataset {self.path} done init")

    # idea to identify dataset file -> use hash over file

    def save(self):
        if not Path(self._cache_dir).exists():
            Path(self._cache_dir).mkdir()
        with open(os.path.join(self._cache_dir, self._cache_file), "wb") as f:
            torch.save(self.data, f)

    def open(self):
        with open(os.path.join(self._cache_dir, self._cache_file), "rb") as f:
            return torch.load(f)

    @staticmethod
    def line2data(line):
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        splitted = line.split("\t")
        x = [float(i) for i in splitted[9:35]]
        # TODO this is still wrong!!!
        y = [float(i) for i in splitted[35:37]]
        expert = float(splitted[6])
        # meta = [float(i) for i in splitted[0:5]]
        return x, y, expert  # , meta

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx):
        if self.in_ram:
            return self.data["x"][idx], self.data["y"][idx]
        else:
            line = linecache.getline(self.path, idx)
            x, y, _, _ = self.line2data(line)
            return torch.Tensor(x), torch.Tensor(y)


class BelleIIExpert(BelleII):
    # DEPCRATED since 0.3
    def __init__(self, path, expert) -> None:
        # only supports in ram
        super().__init__(path, in_ram=True)
        self.expert = expert
        # filter out all samples that do not belong to this expert
        # create index map
        keep = [idx for idx, i in enumerate(
            self.data["expert"]) if i == self.expert]
        # overwrite in order to get back memory from unused data
        self.data = {key: val[keep] for key, val in self.data.items()}
        # senity check
        assert (self.data["expert"] == self.expert).all()

        # set the length correct
        # self.l = len(self.data["x"])
        self.logger.debug(
            f"Dataset {self.path} expert #{self.expert} done init")

# TODO: add preprocessing
class BelleIIBetter(Dataset):
    _cache_dir = ".cache"
    Z_SCALING = [-100, 100]
    THETA_SCALING = [10, 170]
    def __init__(self, path: str, logger: logging.Logger, out_dim: int = 2, compare_to: Optional[str] = None) -> None:
        # out_dim either 2 or 1 if only z should be compared
        super().__init__()
        self.path = path
        self._cache_file = f"{md5(self.path)}.pt"
        self.logger = logger
        self.out_dim = out_dim

        if Path(os.path.join(self._cache_dir, self._cache_file)).exists():
            self.logger.debug("Already cached, loading it")
            self.data = self.open()
        else:
            dt = self.get_data_array()
            self.data = {
                "x": torch.Tensor(dt[:, 9:36]),
                # only 36:37 if only z (out_dim=2)
                "y": torch.Tensor(dt[:, 36:36+out_dim]),
                "expert": torch.Tensor(dt[:, 6]),
                # out_dim==2 -> -4:-1:2 out_dim==1 -> -4:-3:2
                "y_hat_old": torch.Tensor(dt[:, -4:(-1 if out_dim == 2 else -3):2]),
                "idx": torch.arange(dt.shape[0])
            }

            self.save()

        if compare_to:
            # when we want to compare to different predictions
            with open(compare_to, "rb") as f:
                y_hat_old = torch.load(f)
            if out_dim == 1:
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
        super().__init__(*args, **kwargs)
        self.expert = expert
        # filter out all samples that do not belong to this expert
        # create index map
        keep = torch.where(self.data["expert"] == self.expert)
        # overwrite in order to get back memory from unused data
        self.data = {key: val[keep] for key, val in self.data.items()}
        # senity check
        assert (self.data["expert"] == self.expert).all()

class BelleIIBetterExpertDist(BelleIIBetterExpert):
    # TODO: should this return batches?
    # TODO: experts?

    N_BUCKETS = 21
    MEAN = 0
    STD = 0.4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sort_z = [(idx, i[0].item()) for idx, i in enumerate(self.data["y"])]
        self.sort_z = sorted(self.sort_z, key=lambda x: x[1])
        self.buckets = {}
        for idx, z in self.sort_z:
            bucket = self.get_bucket(z)
            if bucket not in self.buckets:
                self.buckets[self.get_bucket(z)] = []

            self.buckets[self.get_bucket(z)].append(idx)

        self.bucket_idx = np.arange(len(self.buckets))
        self.dist = norm(loc=self.MEAN, scale=self.STD)

        # print([self.get_bounds(bucket) for bucket in self.bucket_idx])
        self.probs = [self.get_prob_for_bounds(*self.get_bounds(bucket)) for bucket in self.bucket_idx]
        # self.probs = [1/self.N_BUCKETS for _ in self.bucket_idx] #[self.get_prob_for_bounds(*self.get_bounds(bucket)) for bucket in self.bucket_idx]


    def get_prob_for_bounds(self, lower, upper):
        return self.dist.cdf(upper) - self.dist.cdf(lower)
    

    def get_bounds(self, bucket):
        lower = 2*(bucket/self.N_BUCKETS - 0.5)
        upper = lower + 2/self.N_BUCKETS
        if math.isclose(lower, -1):
            lower = -math.inf
        if math.isclose(upper, 1):
            upper = math.inf
        return lower, upper

    def get_bucket(self, z):
        return math.floor((z/2 + 0.5)*self.N_BUCKETS)

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
