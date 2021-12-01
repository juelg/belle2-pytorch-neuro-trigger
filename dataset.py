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


class BelleII(Dataset):
    def __init__(self, path, in_ram=True) -> None:
        super().__init__()
        # path = /home/tobi/neurotrigger/train1
        self.l = sum(1 for _ in open(path))
        self.path = path
        self.in_ram = in_ram
        self.x, self.y = None, None
        if self.in_ram:
            if Path(self.get_cpath()).exists():
                self.x, self.y = self.open()
            else:
                t1 = time.time()
                with Pool(10) as p:
                    splitted = p.map(self.line2data, open(self.path).readlines())
                # splitted = [self.line2data(line) for line in open(self.path).readlines()]
                t2 = time.time()
                print(t2-t1)

                self.x = torch.Tensor([i[0] for i in splitted])
                self.y = torch.Tensor([i[1] for i in splitted])
                self.save()

    def get_cpath(self, target=False):
        return (".cache/y" if target else ".cache/x") + self.path.split("/")[-1]

    def save(self):
        if not Path(".cache").exists():
            Path(".cache").mkdir()
        with open(self.get_cpath(False), "wb") as f:
            torch.save(self.x, f)
        with open(self.get_cpath(True), "wb") as f:
            torch.save(self.y, f)
                
    def open(self):
        with open(self.get_cpath(False), "rb") as f:
            x = torch.load(f)
        with open(self.get_cpath(True), "rb") as f:
            y = torch.load(f)
        return x, y


    @staticmethod
    def line2data(line):
        splitted = line.split("\t")
        x = [float(i) for i in splitted[8:35]]
        y = [float(i) for i in splitted[35:37]]
        return x, y


    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        if self.in_ram:
            return self.x[idx], self.y[idx]
        else:
            line = linecache.getline(self.path, idx)
            x, y = self.line2data(line)
            return torch.Tensor(x), torch.Tensor(y)



if __name__ == "__main__":
    b = BelleII("/home/tobi/neurotrigger/train1")
    for i in b[:10]:
        print(i)




