from abc import ABC, abstractmethod
import os
from typing import Optional, Tuple
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
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

from neuro_trigger.pytorch.dataset import BelleIIDataset
from pathlib import Path
from pytorch_lightning.loggers.base import DummyLogger


class NTPlot(ABC):
    def __init__(self, vis: "Visualize"):
        self.vis = vis

    @abstractmethod
    def create_plot(self, y: torch.tensor, y_hat: torch.tensor) -> Tuple[Figure, str]:
        """When subclassed this method should create a matplotlib figure and return it together with
        a name. The figure will then be pushed to tensorboard and saved after training in the log folder.

        Args:
            y (torch.tensor): networks output, usally z and theta (however only z if we only train on z)
            y_hat (torch.tensor): wanted output, ground truth

        Returns:
            Tuple[Figure, str]: matplotlib figure and a name identifing the figure
        """
        pass

    def __call__(
        self,
        y: torch.tensor,
        y_hat: torch.tensor,
        suffix: str = "",
        save: Optional[str] = None,
    ):
        """Object of the class can be called like a function. Calls the underlying `create_plot` method
        with the respective arguments and pushes the plot to tensorboard unsing the self.vis.plot method.

        Args:
            y (torch.tensor): networks output, usally z and theta (however only z if we only train on z)
            y_hat (torch.tensor): wanted output, ground truth
            suffix (str, optional): suffix after the identifing name to differentiate between plots with different data. Defaults to "".
            save (Optional[str], optional): Whether the created plot should be saved to the given path. If None the plot won't be saved. Defaults to None.
        """
        fig, name = self.create_plot(y, y_hat)
        if fig == None and name == None:
            return
        self.vis.plot(name + suffix, fig, save=save)


class ZPlot(NTPlot):
    def create_plot(self, y: torch.tensor, y_hat: torch.tensor):
        # scatter histogram
        fig, ax = plt.subplots(dpi=200)
        y = y[:, 0].numpy()
        y_hat = y_hat[:, 0].numpy()
        h = ax.hist2d(y, y_hat, 200, norm=LogNorm(), cmap="jet")

        ax.plot([], [], " ", label=f"Num: {len(y)}")
        ax.plot([], [], " ", label=f"Mean x: {np.mean(y):.{3}f}")
        ax.plot([], [], " ", label=f"Std x: {np.std(y):.{3}f}")
        ax.plot([], [], " ", label=f"Mean y: {np.mean(y_hat):.{3}f}")
        ax.plot([], [], " ", label=f"Std y: {np.std(y_hat):.{3}f}")
        ax.legend()

        cbar = fig.colorbar(h[3], ax=ax)
        # TODO: Fixme, for some reason this latex formula makes problems
        cbar.set_label(r"$\log_{10}$ density of points")

        ax.set(xlabel="reco Track z", ylabel="nnhw Track z", title="z0 reco vs z0 nnhw")
        ax.axis("square")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(BelleIIDataset.Z_SCALING)  # [-100, 100]
        ax.set_ylim(BelleIIDataset.Z_SCALING)

        return fig, "z-plot"


class HistPlot(NTPlot):
    def create_plot(
        self, _: Optional[torch.tensor], y_hat: torch.tensor, xlabel: str = "Neuro Z"
    ):
        # todo adapt labels
        fig, ax = plt.subplots(dpi=200)
        y_hat = y_hat[:, 0].numpy()
        ax.hist(y_hat, bins=100)
        ax.plot([], [], " ", label=f"Num: {len(y_hat)}")
        ax.plot([], [], " ", label=f"Mean: {np.mean(y_hat):.{3}f}")
        ax.plot([], [], " ", label=f"Std: {np.std(y_hat):.{3}f}")
        ax.legend()
        ax.set(xlabel=xlabel)
        return fig, "z-hist"


class DiffPlot(NTPlot):
    def create_plot(self, y: torch.tensor, y_hat: torch.tensor):
        diff = y[:, 0].numpy() - y_hat[:, 0].numpy()
        # entries, std, mean
        fig, ax = plt.subplots(dpi=200)
        ax.hist(diff, bins=100)
        ax.plot([], [], " ", label=f"Num: {len(diff)}")
        ax.plot([], [], " ", label=f"Mean: {np.mean(diff):.{3}f}")
        ax.plot([], [], " ", label=f"Std: {np.std(diff):.{3}f}")
        ax.plot(
            [],
            [],
            " ",
            label=f"Trimmed std: {scipy.stats.mstats.trimmed_std(diff, limits=(0.05, 0.05)):.{3}f}",
        )
        ax.set(xlabel="z(Reco-Neuro)")
        ax.set_xlim(BelleIIDataset.Z_SCALING)
        ax.legend()
        ax.grid()
        return fig, "z-diff"


class ShallowDiffPlot(NTPlot):
    def create_plot(self, y: torch.tensor, y_hat: torch.tensor):
        # +/-1 diff plot -> just limit reco z on pm 1cm
        diff = y[:, 0].numpy() - y_hat[:, 0].numpy()
        diff = diff[list((-1 <= y[:, 0]) & (y[:, 0] <= 1))]
        fig, ax = plt.subplots(dpi=200)

        ax.hist(diff, bins=100)
        ax.plot([], [], " ", label=f"Num: {len(diff)}")
        ax.plot([], [], " ", label=f"Mean: {np.mean(diff):.{3}f}")
        ax.plot([], [], " ", label=f"Std: {np.std(diff):.{3}f}")
        # TODO: some kind of weird warning
        ax.plot(
            [],
            [],
            " ",
            label=f"Trimmed std: {scipy.stats.mstats.trimmed_std(diff, limits=(0.05, 0.05)):.{3}f}",
        )
        ax.set(xlabel="z(Reco-Neuro)")
        ax.legend()
        ax.grid()
        return fig, "z-shallow-diff"


class StdPlot(NTPlot):
    def create_plot(self, y: torch.tensor, y_hat: torch.tensor):
        y, y_hat = y[:50000], y_hat[:50000]
        # TODO: dont use sorted and just plot x'es or dots
        z_diff = y[:, 0].numpy() - y_hat[:, 0].numpy()
        y_sorted = np.sort(y[:, 0])
        y_sorted = y_sorted[(-75 < y_sorted) & (y_sorted < 75)]

        if len(y_sorted) == 0:
            # avoid error in unit tests for empty y_sorted
            return None, None

        def f(yi):
            return scipy.stats.mstats.trimmed_std(
                z_diff[np.array((yi - 1 < y[:, 0]) & (y[:, 0] < yi + 1))],
                limits=(0.05, 0.05),
            )

        stds = np.vectorize(f)(y_sorted[::100])

        fig, ax = plt.subplots(dpi=200)
        ax.plot(y_sorted[::100], stds)
        ax.plot([], [], " ", label=f"Num: {len(z_diff)}")
        ax.plot([], [], " ", label=f"Min: {min(stds):.{3}f}")
        ax.legend()
        ax.grid()
        ax.set(xlabel="reco z")
        ax.set(ylabel="std (reco-neuro)")
        return fig, "z-std"


class Visualize:
    MAX_SAMPLES = 1000000

    def __init__(
        self, module: pl.LightningModule, data: Dataset, folder: Optional[str] = None
    ) -> None:
        self.module = module
        # self.data = Subset(data, np.arange(len(data))[:40000])
        self.data = data
        self.plots = [
            ZPlot(self),
            HistPlot(self),
            DiffPlot(self),
            StdPlot(self),
            ShallowDiffPlot(self),
        ]

        self.should_create_baseline_plots = True

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        y_hat = []
        y = []

        for batch in DataLoader(
            self.data, batch_size=64, num_workers=0, pin_memory=True, drop_last=True
        ):
            x_i, y_i = batch
            y_hat.append(self.module.model(x_i.to(self.module.device)).to("cpu"))
            y.append(y_i)

        y = torch.cat(y)
        y_ = torch.cat(y_hat)
        return y, y_

    def buf2tensor(self, buf: io.BytesIO) -> torch.Tensor:
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        return image

    def fig2buf2tensor(self, fig: Figure) -> torch.Tensor:
        if isinstance(fig, PIL.Image.Image):
            # if is already PIL image
            return ToTensor()(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.buf2tensor(buf)

    def create_baseline_plots(self, save: Optional[str] = None):
        self.create_plots(
            self.data.data["y"], self.data.data["y_hat_old"], suffix="-old", save=save
        )
        y = self.data.data["y"][: self.MAX_SAMPLES]
        y = BelleIIDataset.to_physics(y)

        hp = HistPlot(self)
        suffix = "-gt"
        fig, name = hp.create_plot(None, y, xlabel="Reco Z")
        self.plot(name + suffix, fig, save=save)

    def plot(self, name: str, fig: Figure, save: Optional[str] = None):
        if not save:
            # put figure to tensorboard
            img = self.fig2buf2tensor(fig)
            self.get_tb_logger().experiment.add_image(
                name, img, self.module.current_epoch
            )
        else:
            # save figure to folder
            if not Path(save).exists():
                Path(save).mkdir(parents=True)
            fig.savefig(os.path.join(save, f"{name}.png"), dpi=200, bbox_inches="tight")

    def create_plots(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        suffix: str = "",
        save: Optional[str] = None,
        create_baseline_plots: bool = False,
    ):
        if isinstance(self.module.logger, DummyLogger):
            # this is a test run, dont create plots!
            return
        if (self.module.current_epoch % 10) != 0 and save is None:
            return
        self.module.file_logger.debug(f"Creating plots for expert {self.module.expert}")
        if self.should_create_baseline_plots or create_baseline_plots:
            # create plot once for old nn data
            self.should_create_baseline_plots = False
            self.create_baseline_plots(save=save)
        self.module.file_logger.debug(
            f"Done baseline plots for expert {self.module.expert}"
        )

        y, y_hat = y.cpu()[: self.MAX_SAMPLES], y_hat.cpu()[: self.MAX_SAMPLES]
        y, y_hat = BelleIIDataset.to_physics(y), BelleIIDataset.to_physics(y_hat)
        for plot_f in self.plots:
            plot_f(y, y_hat, suffix, save=save)
            self.module.file_logger.debug(
                f"Done plots {plot_f} for expert {self.module.expert}"
            )
        plt.close("all")

    def get_tb_logger(self) -> TensorBoardLogger:
        if isinstance(self.module.logger, TensorBoardLogger):
            return self.module.logger
        for logger in self.module.logger:
            if isinstance(logger, TensorBoardLogger):
                return logger
        raise RuntimeError("There must be a TensorBoardLogger within the loggers")
