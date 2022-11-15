import torch.nn as nn

from unet.layers.global_pooling import GlobalAvgPool, GlobalMaxPool
from unet.layers.periodic import PeriodicConv1D, PeriodicConv2D


def get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    padding_mode: str = "zeros",
    n_dims: int = 1,
    conv_type: str = "default",
):
    def _get_default_conv():
        if n_dims == 1:
            return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        elif n_dims == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        elif n_dims == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        else:
            raise NotImplementedError("'n_dims' cannot be grater than 3")

    def _get_periodic_conv():
        if n_dims == 1:
            return PeriodicConv1D(
                in_channels, out_channels, kernel_size, stride=stride, padding_mode="circular", padding=padding
            )
        elif n_dims == 2:
            return PeriodicConv2D(
                in_channels, out_channels, kernel_size, stride=stride, padding_mode="circular", padding=padding
            )
        else:
            raise NotImplementedError(f"'n_dims' cannot be grater than 2 for conv_type={conv_type}")

    if conv_type == "default":
        return _get_default_conv()
    elif conv_type == "periodic":
        return _get_periodic_conv()
    else:
        raise NotImplementedError(f"{conv_type} is not supported. Use 'default' or 'periodic' instead")


def get_pooling_layer(
    kernel_size: int = 2,
    stride: int = 2,
    padding: int = 0,
    pool_type: str = "max",
    n_dims: int = 1,
):
    def _get_max_pooling_layer():
        if n_dims == 1:
            return nn.MaxPool1d(kernel_size, stride, padding)
        elif n_dims == 2:
            return nn.MaxPool2d(kernel_size, stride, padding)
        else:
            raise NotImplementedError("'n_dims' cannot be grater than 3")

    def _get_average_pooling_layer():
        if n_dims == 1:
            return nn.AvgPool1d(kernel_size, stride, padding)
        elif n_dims == 2:
            return nn.AvgPool2d(kernel_size, stride, padding)
        else:
            raise NotImplementedError("'n_dims' cannot be grater than 3")

    if pool_type == "max":
        return _get_max_pooling_layer()
    elif pool_type == "avg":
        return _get_average_pooling_layer()
    else:
        raise NotImplementedError


def get_upconv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
    padding: int = 0,
    n_dims: int = 1,
):
    if n_dims == 1:
        return nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
    elif n_dims == 2:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    elif n_dims == 3:
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        raise NotImplementedError("'n_dims' cannot be grater than 3")


def get_global_pooling_layer(
    dim: int = -1,
    type: str = "max",
):
    if type == "max":
        return GlobalMaxPool(dim)
    elif type == "avg":
        return GlobalAvgPool(dim)
    else:
        raise NotImplementedError(f"'{type}' is not supported. Use 'max' or 'avg' instead")
