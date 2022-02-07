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


# TODO: add preprocessing
class BelleII(Dataset):
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



class BelleIIBetter(Dataset):
    _cache_dir = ".cache"
    Z_SCALING = [-100, 100]
    THETA_SCALING = [10, 170]
    def __init__(self, path, logger, out_dim) -> None:
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
            with gzip.open(path, "rt") as f:
                dt = np.loadtxt(path, skiprows=2)
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


        self.logger.debug(
            f"Dataset {self.path} with length {len(self)} done init")
        print("done")

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

    def __getitem__(self, idx):
        return self.data["x"][idx], self.data["y"][idx], self.data["y_hat_old"][idx], self.data["idx"][idx]

    @staticmethod
    def scale(x, lower, upper, lower_new, upper_new):
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
    def __init__(self, expert, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.expert = expert
        # filter out all samples that do not belong to this expert
        # create index map
        keep = torch.where(self.data["expert"] == self.expert)
        # overwrite in order to get back memory from unused data
        self.data = {key: val[keep] for key, val in self.data.items()}
        # senity check
        assert (self.data["expert"] == self.expert).all()

        self.logger.debug(
            f"Dataset {self.path} expert #{self.expert} with length {len(self)} done init")


if __name__ == "__main__":
    b = BelleIIBetter(
        "/remote/neurobelle/data/dqmNeuro/dqmNeuro_mpp34_exp20_430-459/lt100reco/idhist_10170_default/section_fp/neuroresults_random1.gz", logging.getLogger("test"))
    for i in b[:1]:
        print(i)
