import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from matplotlib.colors import LogNorm
import io
from torchvision.transforms import ToTensor
import PIL
import torch

# predict # amount of val -> plot z (first coord) agains z true


class Visualize:
    def __init__(self, module, data) -> None:
        self.module = module
        self.data = Subset(data, np.arange(len(data))[:40000])


    def forward(self):
        y_ = []
        y = []

        for batch in DataLoader(self.data, batch_size=64, num_workers=0, pin_memory=True, drop_last=True):
            x_i, y_i = batch
            y_.append(self.module.model(x_i.to(self.module.device)).to("cpu"))
            y.append(y_i)

        y = torch.cat(y)
        y_ = torch.cat(y_)
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


    def create_plot(self):
        y, y_ = self.forward()
        # def s(x, y, intensity=True, x_lim=None, y_lim=None, xlabel=None, ylabel=None, title=None, nbuckets=500):

        # scatter histogram
        fig, ax = plt.subplots(dpi=300)
        h = ax.hist2d(y[:,0].numpy(),y_[:,0].numpy(), 200, norm=LogNorm(),cmap='jet')
        
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label(r'$\log_{10}$ density of points')

        ax.set(xlabel="nnhw Track z[cm]", ylabel="reco Track z[cm]", title="z0 reco vs z0 nnhw")
        ax.axis('square')
        ax.set_aspect('equal', adjustable='box')

        # if x_lim:
        #     ax.set_xlim(x_lim)
        # if y_lim:
        #     ax.set_ylim(y_lim)
        # plt.show()

        # do something with fig
        img = self.fig2buf2tensor(fig)
        self.module.logger.experiment.add_image("z-plot", img, self.module.current_epoch)


