from torch.nn.modules.loss import _Loss
from SphericalHarmonicTransformation import SH2Signal
import torch


class MSESignal(_Loss):
    """
    MSESignal(output, label, mask) -> loss

    calculates the MSE for a spherical signal given in spherical harmonic domain in signal domain.

    Args:
        output  (tensor, float): networks output tensor
                (batchsize x number of shells * number of gradients x dim_x x dim_y x dim_z)
        label   (tensor, float): given label tensor
                (batchsize x number of shells * number of gradients x dim_x x dim_y x dim_z)
        mask    (tensor, long): binary mask for with given evaluation mask (e.g. white matter only)
                (batchsize x dim_x x dim_y x dim_z)
    Example::

        loss = MSESignal(sh_order=4, gradient=np.ones((30, 3)))

        output = torch.zeros((128, 15*3, 5, 5, 5))
        label = torch.ones((128, 15*3, 5, 5, 5))
        mask = torch.ones((128, 5, 5, 5)).long()
        loss(output, label, mask)
    """
    def __init__(self, sh_order, gradients, no_average=False):
        super(MSESignal, self).__init__()
        self.sh_order = sh_order
        self.sh2signal = SH2Signal(sh_order, gradients)
        self.no_average = no_average

    def forward(self, output, label, mask):
        tmp_output = self.sh2signal(output)
        tmp_label = self.sh2signal(label)

        mse = torch.mean((tmp_output-tmp_label)**2, 1).masked_select(mask.byte())
        if not self.no_average:
            mse = torch.mean(mse)
        return mse

