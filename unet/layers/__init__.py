import torch.nn as nn

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
        else:
            raise NotImplementedError

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
            raise NotImplementedError

    if conv_type == "default":
        return _get_default_conv()
    elif conv_type == "periodic":
        return _get_periodic_conv()
    else:
        raise NotImplementedError
