from scipy import special as sci
import torch.nn as nn
import numpy as np
import torch
from DELIMIT.utils import sph2cart, cart2sph


class Signal2SH(nn.Module):
    """
    Signal2SH(dwi) -> dwi_sh

    Computes the corresponding spherical harmonic coefficients

    Args:
        x_in (5D tensor): input dwi tensor
        x_in.size(): (Batchsize x Number of shells * Number of gradients x DimX x DimY x DimZ)
        y (5D tensor): corresponding harmonic coefficients tensor
        y.size(): (Batchsize x Number of shells*Number of coefficients x DimX x DimY x DimZ)
    """

    def __init__(self, sh_order, gradients, lb_lambda=0.006):
        super(Signal2SH, self).__init__()
        self.sh_order = sh_order
        self.lb_lambda = lb_lambda
        self.num_gradients = gradients.shape[0]
        self.num_coefficients = int((self.sh_order + 1) * (self.sh_order / 2 + 1))

        b = np.zeros((self.num_gradients, self.num_coefficients))
        l = np.zeros((self.num_coefficients, self.num_coefficients))
        for id_gradient in range(self.num_gradients):
            id_column = 0
            for id_order in range(0, self.sh_order + 1, 2):
                for id_degree in range(-id_order, id_order + 1):
                    gradients_phi, gradients_theta, gradients_z = cart2sph(gradients[id_gradient, 0],
                                                                           gradients[id_gradient, 1],
                                                                           gradients[id_gradient, 2])
                    y = sci.sph_harm(np.abs(id_degree), id_order, gradients_phi, gradients_theta)

                    if id_degree < 0:
                        b[id_gradient, id_column] = np.real(y) * np.sqrt(2)
                    elif id_degree == 0:
                        b[id_gradient, id_column] = np.real(y)
                    elif id_degree > 0:
                        b[id_gradient, id_column] = np.imag(y) * np.sqrt(2)

                    l[id_column, id_column] = self.lb_lambda * id_order ** 2 * (id_order + 1) ** 2
                    id_column += 1

        b_inv = np.linalg.pinv(np.matmul(b.transpose(), b) + l)
        self.Signal2SHMat = torch.nn.Parameter(torch.from_numpy(np.matmul(b_inv, b.transpose()).transpose()).float(),
                                               requires_grad=False)

    def forward(self, x_in):
        x = x_in.reshape((-1, np.ceil(x_in.size(1) / self.num_gradients).astype(int), self.num_gradients, x_in.size(2),
                          x_in.size(3), x_in.size(4)))
        x = x.permute(0, 1, 3, 4, 5, 2)
        y = x.matmul(self.Signal2SHMat)
        y = y.permute(0, 1, 5, 2, 3, 4).contiguous().reshape((x.size(0), -1, x_in.size(2), x_in.size(3), x_in.size(4)))
        return y


class SH2Signal(nn.Module):
    """
    SH2Signal(dwi_sh) -> dwi

    Computes the corresponding dwi signal for each gradient

    Args:
        x_in (5D tensor): input spherical harmonic tensor
        x_in.size(): (Batchsize x Number of shells*Number of coefficients x DimX x DimY x DimZ)
        y (5D tensor): corresponding dwi tensor
        y.size(): (Batchsize x Number of shells * Number of gradients x DimX x DimY x DimZ)
    """

    def __init__(self, sh_order, gradients):
        super(SH2Signal, self).__init__()
        self.sh_order = sh_order
        self.num_gradients = gradients.shape[0]
        self.num_coefficients = int((self.sh_order + 1) * (self.sh_order / 2 + 1))

        SH2SignalMat = np.zeros((self.num_coefficients, self.num_gradients))
        for id_gradient in range(self.num_gradients):
            id_coefficient = 0
            for id_order in range(0, self.sh_order + 1, 2):  # even order only
                for id_degree in range(-id_order, id_order + 1):
                    gradients_phi, gradients_theta, gradients_z = cart2sph(gradients[id_gradient, 0],
                                                                           gradients[id_gradient, 1],
                                                                           gradients[id_gradient, 2])
                    y = sci.sph_harm(np.abs(id_degree), id_order, gradients_phi, gradients_theta)
                    if id_degree < 0:
                        SH2SignalMat[id_coefficient, id_gradient] = np.real(y) * np.sqrt(2)
                    elif id_degree == 0:
                        SH2SignalMat[id_coefficient, id_gradient] = np.real(y)
                    elif id_degree > 0:
                        SH2SignalMat[id_coefficient, id_gradient] = np.imag(y) * np.sqrt(2)

                    id_coefficient += 1

        self.SH2SignalMat = torch.nn.Parameter(torch.from_numpy(SH2SignalMat).float(), requires_grad=False)

    def forward(self, x_in):
        x_dim = x_in.size()
        x = x_in.reshape((x_dim[0], np.ceil(x_in.size(1) / self.num_coefficients).astype(int), self.num_coefficients,
                          x_dim[-3], x_dim[-2], x_dim[-1]))
        x = x.permute(0, 1, 3, 4, 5, 2)
        y = x.matmul(self.SH2SignalMat)
        y = y.permute(0, 1, 5, 2, 3, 4).contiguous().reshape((x_dim[0], -1, x_dim[-3], x_dim[-2], x_dim[-1]))
        return y

