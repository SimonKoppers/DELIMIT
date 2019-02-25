from torch.nn.modules.loss import _Loss
from SphericalHarmonicTransformation import SH2Signal
import torch


class MSESignal(_Loss):
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

