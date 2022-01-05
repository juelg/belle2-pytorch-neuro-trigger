from typing import Tuple
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


class Visualize:
    MAX_SAMPLES = 1000000

    def __init__(self, module, data) -> None:
        self.module = module
        # self.data = Subset(data, np.arange(len(data))[:40000])
        self.data = data
        self.plots = [self.z_plot, self.hist_plot]

        self.should_create_baseline_plots = True

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
        self.hist_plot(None, self.data.data["y"], suffix="-gt")

    def create_plots(self, y: torch.Tensor, y_hat: torch.Tensor, suffix=""):
        if self.should_create_baseline_plots:
            # create plot once for old nn data
            self.should_create_baseline_plots = False
            self.create_baseline_plots()

        y, y_hat = y.cpu()[:self.MAX_SAMPLES], y_hat.cpu()[:self.MAX_SAMPLES]
        for plot_f in self.plots:
            plot_f(y, y_hat, suffix)

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
        h = ax.hist2d(y[:, 0].numpy(), y_hat[:, 0].numpy(),
                      200, norm=LogNorm(), cmap='jet')

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label(r'$\log_{10}$ density of points')

        ax.set(xlabel="reco Track z", ylabel="nnhw Track z",
               title="z0 reco vs z0 nnhw")
        ax.axis('square')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

        img = self.fig2buf2tensor(fig)
        self.get_tb_logger().experiment.add_image(
            f"z-plot{suffix}", img, self.module.current_epoch)

    def hist_plot(self, y, y_hat, suffix=""):
        fig, ax = plt.subplots(dpi=200)
        ax.hist(y_hat[:, 0].numpy(), bins=100)
        ax.set(xlabel="Neuro Z")
        img = self.fig2buf2tensor(fig)
        self.get_tb_logger().experiment.add_image(
            f"z-hist{suffix}", img, self.module.current_epoch)
