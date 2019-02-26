import torch
import torch.nn as nn
import numpy as np
from DELIMIT.utils import sph2cart
from DELIMIT.SphericalHarmonicTransformation import SH2Signal, Signal2SH
from pyquaternion import Quaternion


class LocalSphericalConvolution(nn.Module):
    """
    defines the LocalSphericalConvolution as a torch module, which can be utilized within a DL network

    Args:
        shells_in  (int):                   number of input shells
        shells_out  (int):                  number of output shells
        sh_order_in  (int):                 Spherical Harmonic order of the input signal
        sh_order_out  (int):                Spherical Harmonic order of the output signal
        sampled_gradients  (np.array, Nx3): gradients that are given by the signal, or shall be used during application
        kernel_sizes (list):                list of number of sampled points arround each sampled gradient point.
                                            Each entry defines a new angular circle around the gradient.
        lb_lambda (float):                  laplace beltrami regularization during SH fit (e.g. 0.006 ).

    Example::

        lsc = LocalSphericalConvolution(shells_in=3, shells_out=128,
                                        sh_order_in=4, sh_order_out=8, lb_lambda=0.006,
                                        sampled_gradients=np.ones((30, 3)), kernel_sizes=[5, 5],
                                        angular_distance=(np.pi / 10))

        input_tensor = torch.zeros((128, 15*3, 5, 5, 5))
        output_tensor = lsc(input_tensor)
    """

    def __init__(self, shells_in, shells_out, sh_order_in, sh_order_out, sampled_gradients, kernel_sizes,
                 lb_lambda=0.006, angular_distance=0):
        super(LocalSphericalConvolution, self).__init__()
        self.shells_in = shells_in
        self.shells_out = shells_out

        self.sh_order_in = sh_order_in
        self.coeff_in = int((self.sh_order_in + 1) * (self.sh_order_in / 2 + 1))
        self.sh_order_out = sh_order_out
        self.coeff_out = int((self.sh_order_out + 1) * (self.sh_order_out / 2 + 1))

        self.gradients = sampled_gradients
        self.kernel_sizes = kernel_sizes
        self.lb_lambda = lb_lambda

        self.num_gradients = sampled_gradients.shape[0]
        self.kernel_sum = np.asscalar(np.sum(self.kernel_sizes) + 1)

        if angular_distance == 0:
            bvec_mat = np.abs(np.matmul(sampled_gradients, sampled_gradients.T))
            bvec_mat[bvec_mat > 0.999] = 0
            angular_distance = np.mean(np.min(np.arccos(bvec_mat), axis=1))
            self.angular_distance = angular_distance
            print('Setting angular distance to: ' + str(angular_distance / np.pi * 180))
        else:
            self.angular_distance = angular_distance

            # define full sampled gradient set
        base_kernel = np.zeros((self.kernel_sum, 3))
        base_kernel[:, 2] = 1
        gradient_id = 1
        for id_layer, num_samples in enumerate(self.kernel_sizes):
            base_kernel[gradient_id:gradient_id + num_samples, 0] = np.arange(-np.pi, np.pi,
                                                                              2 * np.pi / num_samples) + id_layer % 2 * np.pi
            base_kernel[gradient_id:gradient_id + num_samples, 1] = angular_distance * (id_layer + 1)
            gradient_id += num_samples

        base_kernel[:, 0], base_kernel[:, 1], base_kernel[:, 2] = sph2cart(base_kernel[:, 0],
                                                                           base_kernel[:, 1],
                                                                           base_kernel[:, 2])

        # create rotation angle and rotation axis
        rot_angle = np.matmul(np.array([0, 0, 1]), sampled_gradients.T)
        rot_angle = np.arccos(rot_angle) * np.sign(rot_angle)
        rot_axis = np.cross(sampled_gradients, np.array([0, 0, 1]))

        # create new gradient table
        tmp_gradients = np.zeros((sampled_gradients.shape[0] * base_kernel.shape[0], 3))
        for id_gradient in np.arange(sampled_gradients.shape[0]):
            for id_inner_gradient in np.arange(base_kernel.shape[0]):
                tmp_gradients[id_gradient * base_kernel.shape[0] + id_inner_gradient, :] = Quaternion(
                    axis=rot_axis[id_gradient, :], angle=rot_angle[id_gradient]).rotate(
                    base_kernel[id_inner_gradient, :])

        self.conv_gradients = tmp_gradients

        # setup convolutional kernel
        self.sconv = nn.Conv2d(self.shells_in, self.shells_out, kernel_size=(1, self.kernel_sum), bias=True)

        # setup harmonic interpolation layer
        self.signal2sh = Signal2SH(self.sh_order_out, self.gradients, self.lb_lambda)
        self.sh2signal = SH2Signal(self.sh_order_in, self.conv_gradients)

    def forward(self, x_in):
        x_dim = x_in.size()
        x_conv = self.sh2signal(x_in)
        x_conv = x_conv.permute(0, 2, 3, 4, 1).contiguous()
        x_conv = x_conv.view(-1, self.shells_in, self.num_gradients, self.kernel_sum)
        x_conv = self.sconv(x_conv)
        x_conv = x_conv.view(x_dim[0], x_dim[-3], x_dim[-2], x_dim[-1], self.shells_out * self.num_gradients)
        x_conv = x_conv.permute(0, 4, 1, 2, 3)
        x_out = self.signal2sh(x_conv)
        return x_out


class SphericalConvolution(nn.Module):
    """
    defines the SphericalConvolution as a torch module, which can be utilized within a DL network

    Args:
        shells_in  (int):                   number of input shells
        shells_out  (int):                  number of output shells
        sh_order  (int):                    utilzed Spherical Harmonic order

    Example::

        sc = SphericalConvolution(shells_in=3, shells_out=128, sh_order=8)
        input_tensor = torch.zeros((128, 45*3, 5, 5, 5))
        output_tensor = lsc(input_tensor)
    """

    def __init__(self, shells_in, shells_out, sh_order):
        super(SphericalConvolution, self).__init__()
        self.shells_in = shells_in
        self.shells_out = shells_out

        self.sh_order = sh_order
        self.ncoeff = int((self.sh_order + 1) * (self.sh_order / 2 + 1))
        self.weight = nn.Parameter(torch.randn((shells_out, shells_in, int(self.sh_order / 2) + 1)))
        self.bias = nn.Parameter(torch.zeros((shells_out, 1)))

        sh_indices = torch.zeros(self.ncoeff).long()
        last_coeff = 0
        for id_order in range(0, int(self.sh_order / 2) + 1):
            sh_order = id_order * 2
            ncoeff = int((sh_order + 1) * (sh_order / 2 + 1))
            sh_indices[last_coeff:ncoeff] = id_order
            last_coeff = ncoeff

        self.sh_indices = torch.nn.Parameter(sh_indices.long(), requires_grad=False)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.bias, mode='fan_out', nonlinearity='relu')

    def forward(self, x_in):
        x_dim = x_in.size()
        x = x_in.permute(0, 2, 3, 4, 1)
        x = x.reshape(-1, self.shells_in, self.ncoeff)
        kernel = torch.index_select(self.weight, -1, self.sh_indices)

        x = torch.mean(torch.mul(x.unsqueeze(1), kernel), dim=2)
        x = x + self.bias.unsqueeze(0)

        x = x.reshape(x_dim[0], x_dim[-3], x_dim[-2], x_dim[-1], self.shells_out * self.ncoeff)
        x_out = x.permute(0, 4, 1, 2, 3)
        return x_out

