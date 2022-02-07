from functools import partial
import multiprocessing
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from matplotlib.colors import LogNorm
import io
from torchvision.transforms import ToTensor
import PIL
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import scipy.stats
from torch.utils.data import Dataset
import pytorch_lightning as pl

from dataset import BelleIIBetter
from pathlib import Path



class Visualize:
    MAX_SAMPLES = 1000000

    def __init__(self, module: pl.LightningModule, data: Dataset, folder: Optional[str]=None) -> None:
        self.module = module
        # self.data = Subset(data, np.arange(len(data))[:40000])
        self.data = data
        self.plots = [self.z_plot, self.hist_plot, self.diff_plot, self.std_plot, self.shallow_diff_plot]

        self.should_create_baseline_plots = True
        # if folder is none than we try to log tensorboard
        self._folder = None
        self.folder = folder

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        self._folder = value
        if self.folder and not Path(self.folder).exists():
            Path(self.folder).mkdir(parents=True)




    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        y_hat = []
        y = []

        for batch in DataLoader(self.data, batch_size=64, num_workers=0, pin_memory=True, drop_last=True):
            x_i, y_i = batch
            y_hat.append(self.module.model(
                x_i.to(self.module.device)).to("cpu"))
            y.append(y_i)

        y = torch.cat(y)
        y_ = torch.cat(y_hat)
        return y, y_

    def buf2tensor(self, buf):
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        return image

    def fig2buf2tensor(self, fig):
        if isinstance(fig, PIL.Image.Image):
            # if is already PIL image
            return ToTensor()(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return self.buf2tensor(buf)

    def create_baseline_plots(self):
        self.create_plots(
            self.data.data["y"], self.data.data["y_hat_old"], suffix="-old")
        y = self.data.data["y"][:self.MAX_SAMPLES]
        y = BelleIIBetter.to_physics(y)
        self.hist_plot(None, y, suffix="-gt", xlabel="Reco Z")

    def plot(self, name, fig):
        if not self.folder:
            # put figure to tensorboard
            img = self.fig2buf2tensor(fig)
            self.get_tb_logger().experiment.add_image(
                name, img, self.module.current_epoch)
        else:
            # save figure to folder
            fig.savefig(os.path.join(self.folder, f"{name}.png"), dpi=200, bbox_inches='tight')

    # def plot(self, fig, name, option="tensorboard"):
    #     if option=="tensorboard":
    #         img = self.fig2buf2tensor(fig)
    #         self.get_tb_logger().experiment.add_image(
    #             name, img, self.module.current_epoch)
    #     elif option == "show":
    #         plt.show()
    #     elif option == "save":
    #         # TODO bound tight usw, special path
    #         plt.savefig(f"{name}.png")

    def create_plots(self, y: torch.Tensor, y_hat: torch.Tensor, suffix=""):
        # TODO add log here to see when stuff is plotted
        if (self.module.current_epoch % 10) != 0:
            return
        if self.should_create_baseline_plots:
            # create plot once for old nn data
            self.should_create_baseline_plots = False
            self.create_baseline_plots()

        y, y_hat = y.cpu()[:self.MAX_SAMPLES], y_hat.cpu()[:self.MAX_SAMPLES]
        y, y_hat = BelleIIBetter.to_physics(y), BelleIIBetter.to_physics(y_hat)
        for plot_f in self.plots:
            plot_f(y, y_hat, suffix)
        plt.close('all')

    def get_tb_logger(self):
        if isinstance(self.module.logger, TensorBoardLogger):
            return self.module.logger
        for logger in self.module.logger:
            if isinstance(logger, TensorBoardLogger):
                return logger
        raise RuntimeError("There must be a TensorBoardLogger within the loggers")



    def z_plot(self, y, y_hat, suffix=""):

        # scatter histogram
        fig, ax = plt.subplots(dpi=200)
        y = y[:, 0].numpy()
        y_hat = y_hat[:, 0].numpy()
        h = ax.hist2d(y, y_hat,
                      200, norm=LogNorm(), cmap='jet')
        
        ax.plot([], [], ' ', label=f"Num: {len(y)}")
        ax.plot([], [], ' ', label=f"Mean x: {np.mean(y):.{3}f}")
        ax.plot([], [], ' ', label=f"Std x: {np.std(y):.{3}f}")
        ax.plot([], [], ' ', label=f"Mean y: {np.mean(y_hat):.{3}f}")
        ax.plot([], [], ' ', label=f"Std y: {np.std(y_hat):.{3}f}")
        ax.legend()

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label(r'$\log_{10}$ density of points')

        ax.set(xlabel="reco Track z", ylabel="nnhw Track z",
               title="z0 reco vs z0 nnhw")
        ax.axis('square')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(BelleIIBetter.Z_SCALING) # [-100, 100]
        ax.set_ylim(BelleIIBetter.Z_SCALING)

        # img = self.fig2buf2tensor(fig)
        # self.get_tb_logger().experiment.add_image(
        #     f"z-plot{suffix}", img, self.module.current_epoch)
        self.plot(f"z-plot{suffix}", fig)

    def hist_plot(self, y, y_hat, suffix="", xlabel="Neuro Z"):
        # todo adapt labels
        fig, ax = plt.subplots(dpi=200)
        y_hat = y_hat[:, 0].numpy()
        ax.hist(y_hat, bins=100)
        ax.plot([], [], ' ', label=f"Num: {len(y_hat)}")
        ax.plot([], [], ' ', label=f"Mean: {np.mean(y_hat):.{3}f}")
        ax.plot([], [], ' ', label=f"Std: {np.std(y_hat):.{3}f}")
        ax.legend()

        # ax.set(xlabel=xlabel)
        # img = self.fig2buf2tensor(fig)
        # self.get_tb_logger().experiment.add_image(
        #     f"z-hist{suffix}", img, self.module.current_epoch)
        self.plot(f"z-hist{suffix}", fig)
    

    def diff_plot(self, y, y_hat, suffix=""):
        diff = y[:, 0].numpy() - y_hat[:, 0].numpy()
        # entries, std, mean
        fig, ax = plt.subplots(dpi=200)

        ax.hist(diff, bins=100)
        ax.plot([], [], ' ', label=f"Num: {len(diff)}")
        ax.plot([], [], ' ', label=f"Mean: {np.mean(diff):.{3}f}")
        ax.plot([], [], ' ', label=f"Std: {np.std(diff):.{3}f}")
        ax.plot([], [], ' ', label=f"Trimmed std: {scipy.stats.mstats.trimmed_std(diff, limits=(0.05, 0.05)):.{3}f}")
        ax.set(xlabel="z(Reco-Neuro)")
        ax.set_xlim(BelleIIBetter.Z_SCALING)
        ax.legend()
        ax.grid()
        # img = self.fig2buf2tensor(fig)
        # self.get_tb_logger().experiment.add_image(
        #     f"z-diff{suffix}", img, self.module.current_epoch)
        self.plot(f"z-diff{suffix}", fig)

    def shallow_diff_plot(self, y, y_hat, suffix=""):
        # +/-1 diff plot -> just limit reco z on pm 1cm
        diff = y[:, 0].numpy() - y_hat[:, 0].numpy()
        diff = diff[(-1 <= y[:,0]) & (y[:,0] <= 1)]
        fig, ax = plt.subplots(dpi=200)

        ax.hist(diff, bins=100)
        ax.plot([], [], ' ', label=f"Num: {len(diff)}")
        ax.plot([], [], ' ', label=f"Mean: {np.mean(diff):.{3}f}")
        ax.plot([], [], ' ', label=f"Std: {np.std(diff):.{3}f}")
        ax.plot([], [], ' ', label=f"Trimmed std: {scipy.stats.mstats.trimmed_std(diff, limits=(0.05, 0.05)):.{3}f}")
        ax.set(xlabel="z(Reco-Neuro)")
        ax.legend()
        ax.grid()
        # img = self.fig2buf2tensor(fig)
        # self.get_tb_logger().experiment.add_image(
        #     f"z-shallow-diff{suffix}", img, self.module.current_epoch)
        self.plot(f"z-shallow-diff{suffix}", fig)

    # TODO, how does this work
    # shwo picture
    def std_plot_old(self, y, y_hat, suffix=""):
        z_diff = y[:, 0].numpy() - y_hat[:, 0].numpy()
        bucket_len = 500
        # could also be solved with zip
        z_sorted = np.zeros((len(z_diff), 2))
        z_sorted[:,0] = z_diff
        z_sorted[:,1] = y[:,0]
        z_sorted = np.array(sorted(z_sorted, key=lambda x: x[0]))
        # always take 20 around, and start at 10 and end at len()-10
        stds = []
        buck = []
        # half_bucket_len = int(bucket_len/2)
        # for i in range(len(z_sorted))[half_bucket_len:-half_bucket_len]:
        #     buck.append(z_sorted[i,1])
        #     stds.append(scipy.stats.mstats.trimmed_std(z_sorted[i-half_bucket_len:i+half_bucket_len][0], limits=(0.05, 0.05)))

        z_sorted = np.array(sorted(z_sorted, key=lambda x: x[1]))
        for diff, y_ in z_sorted:
            buck.append(y)
            stds.append(np.std(z_diff[(y_-1 < y[:,0]) & (y[:,0] < y_+1)]))

        # xy = list(zip(buck, stds))
        # xy = sorted(xy, key=lambda x:x[0])
        # xy = np.array(xy)

        fig, ax = plt.subplots(dpi=200)
        # ax.plot(xy[:,0], xy[:,1])
        ax.plot(buck, stds)
        ax.plot([], [], ' ', label=f"Num: {len(z_diff)}")
        ax.plot([], [], ' ', label=f"Min: {min(stds):.{3}f}")
        ax.plot([], [], ' ', label=f"Bucket len: {bucket_len}")
        ax.legend()
        ax.grid()
        ax.set(xlabel="std z(Reco-Neuro)")
        # img = self.fig2buf2tensor(fig)
        # self.get_tb_logger().experiment.add_image(
        #     f"z-std{suffix}", img, self.module.current_epoch)
        self.plot(f"z-std{suffix}", fig)





    def std_plot(self, y, y_hat, suffix=""):
        y, y_hat = y[:50000], y_hat[:50000]
        # TODO: dont use sorted and just plot x'es or dots
        z_diff = y[:, 0].numpy() - y_hat[:, 0].numpy()
        # stds = []
        # buck = []
        y_sorted = np.sort(y[:,0]) # np.array(sorted(y[:,0]))
        y_sorted = y_sorted[(-75 < y_sorted) & (y_sorted < 75)]

        def f(yi):
            return scipy.stats.mstats.trimmed_std(z_diff[(yi-1 < y[:,0]) & (y[:,0] < yi+1)], limits=(0.05, 0.05))

        # import time
        # t1 = time.time()
        stds = np.vectorize(f)(y_sorted[::100])
        # print(time.time()-t1)


        # t1 = time.time()
        # for yi in y_sorted:
        #     # buck.append(yi)
        #     stds.append(np.std(z_diff[(yi-1 < y[:,0]) & (y[:,0] < yi+1)])) #, limits=(0.05, 0.05)))
        #     # scipy.stats.mstats.trimmed_std
        # print(time.time()-t1)

        # t1 = time.time()
        # with multiprocessing.Pool(64) as p:
        #     asdf = p.map(partial(f2, z_diff, y), y_sorted)
        # print(time.time()-t1)

        fig, ax = plt.subplots(dpi=200)
        ax.plot(y_sorted[::100], stds)
        ax.plot([], [], ' ', label=f"Num: {len(z_diff)}")
        ax.plot([], [], ' ', label=f"Min: {min(stds):.{3}f}")
        ax.legend()
        ax.grid()
        ax.set(xlabel="reco z")
        ax.set(ylabel="std (reco-neuro)")
        # img = self.fig2buf2tensor(fig)
        # self.get_tb_logger().experiment.add_image(
        #     f"z-std{suffix}", img, self.module.current_epoch)
        self.plot(f"z-std{suffix}", fig)

    def std_plot2(self, y, y_hat, suffix=""):
        z_diff = y[:, 0].numpy() - y_hat[:, 0].numpy()
        bucket_len = 5000
        stds = []
        buck = []
        # y_sorted = np.array(sorted(y[:,0]))
        # y_sorted = y_sorted[(-75 < y_sorted) & (y_sorted < 75)]

        y_sorted = np.zeros((len(z_diff), 2))
        y_sorted[:,0] = z_diff
        y_sorted[:,1] = y[:,0]
        y_sorted = np.array(sorted(y_sorted, key=lambda x: x[1]))

        # for yi in y_sorted:
        #     buck.append(yi)
        #     stds.append(np.std(z_diff[(yi-1 < y[:,0]) & (y[:,0] < yi+1)]))

        half_bucket_len = int(bucket_len/2)
        for i in range(len(y_sorted))[half_bucket_len:-half_bucket_len]:
            buck.append(y_sorted[i][1])
            stds.append(scipy.stats.mstats.trimmed_std(y_sorted[i-half_bucket_len:i+half_bucket_len][0], limits=(0.05, 0.05)))

        fig, ax = plt.subplots(dpi=200)
        ax.plot(buck, stds)
        ax.plot([], [], ' ', label=f"Num: {len(z_diff)}")
        ax.plot([], [], ' ', label=f"Min: {min(stds):.{3}f}")
        ax.legend()
        ax.grid()
        ax.set(xlabel="reco z")
        ax.set(ylabel="std (reco-neuro)")
        # img = self.fig2buf2tensor(fig)
        # self.get_tb_logger().experiment.add_image(
        #     f"z-std2{suffix}", img, self.module.current_epoch)
        self.plot(f"z-std2{suffix}", fig)



def f2(z_diff, y, yi):
    return np.std(z_diff[(yi-1 < y[:,0]) & (y[:,0] < yi+1)]) #, limits=(0.05, 0.05)))