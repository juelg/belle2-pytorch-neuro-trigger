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


class Visualize:
    def __init__(self, module, data) -> None:
        self.module = module
        self.data = Subset(data, np.arange(len(data))[:40000])
        self._y = None


    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        y_ = []
        y = []

        for batch in DataLoader(self.data, batch_size=64, num_workers=0, pin_memory=True, drop_last=True):
            x_i, y_i = batch
            y_.append(self.module.model(x_i.to(self.module.device)).to("cpu"))
            y.append(y_i)

        y = torch.cat(y)
        y_ = torch.cat(y_)
        return y, y_

    @property
    def y(self):
        if self._y is None:
            self.y = self.forward()
        return self._y

    @y.setter
    def y(self, y: Tuple[torch.Tensor, torch.Tensor]):
        self._y = (y[0].cpu(), y[1].cpu())


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


    def create_plots(self, batch, y_hat):
        batch = batch.cpu()
        # TODO: call all plot functions, in the first time call them also with old y_hat data to have baseline plots -> do this in init
        # TODO: create ground truth hist once in the beginning


    def z_plot(self):
        y, y_hat = self.y

        # scatter histogram
        fig, ax = plt.subplots(dpi=200)
        h = ax.hist2d(y[:,0].numpy(),y_hat[:,0].numpy(), 200, norm=LogNorm(),cmap='jet')
        
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label(r'$\log_{10}$ density of points')

        ax.set(xlabel="reco Track z", ylabel="nnhw Track z", title="z0 reco vs z0 nnhw")
        ax.axis('square')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

        img = self.fig2buf2tensor(fig)
        self.module.logger.experiment.add_image("z-plot", img, self.module.current_epoch)

    def hist_plot(self):
        y, y_hat = self.y
        fig, ax = plt.subplots(dpi=200)
        ax.hist(y_hat[:, 0].numpy(), bins=100)
        ax.set(xlabel="Neuro Z")
        img = self.fig2buf2tensor(fig)
        self.module.logger.experiment.add_image("z-hist", img, self.module.current_epoch)



