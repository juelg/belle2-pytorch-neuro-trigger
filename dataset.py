from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
from PIL import Image
from PIL import ImageOps
import torch
import random
import linecache
from multiprocessing import Pool
import time
from pathlib import Path
import os
import logging

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
        with open(path) as f:
            self.l = sum(1 for _ in f)
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
                self.logger.debug(f"{self.path} not cached yet, start caching, this might take a while")
                t1 = time.time()
                with Pool(10) as p:
                    with open(self.path) as f:
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
                    #"meta": torch.Tensor([i[3] for i in splitted]),
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
        splitted = line.split("\t")
        x = [float(i) for i in splitted[8:35]]
        y = [float(i) for i in splitted[35:37]]
        expert = float(splitted[5])
        # meta = [float(i) for i in splitted[0:5]]
        return x, y , expert #, meta


    def __len__(self):
        return self.l

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
        keep = [idx for idx, i in enumerate(self.data["expert"]) if i==self.expert]
        # overwrite in order to get back memory from unused data
        self.data = {key: val[keep] for key, val in self.data.items()}
        # senity check
        assert (self.data["expert"] == self.expert).all()

        # set the length correct
        self.l = len(self.data["x"])
        self.logger.debug(f"Dataset {self.path} expert #{self.expert} done init")




if __name__ == "__main__":
    b = BelleII("/home/tobi/neurotrigger/train1")
    for i in b[:10]:
        print(i)




