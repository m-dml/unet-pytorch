import torch
import torch.nn as nn

from unet.layers import get_conv_layer, get_pooling_layer, get_upconv_layer
from unet.model.abstract_blocks import AbstractDecoderBlock, AbstractEncoderBlock, AbstractOutputBlock


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: list,
        kernel_size: int,
        stride: int,
        padding: int,
        padding_mode: str = "zeros",
        conv_type: str = "default",
        activation=nn.ReLU,
        n_dims: int = 1,
    ):
        super(ConvBlock, self).__init__()

        layers = [
            get_conv_layer(in_channels, filters[0], kernel_size, stride, padding, padding_mode, n_dims, conv_type),
            activation(),
        ]
        for i in range(1, len(filters)):
            layers.append(
                get_conv_layer(
                    filters[i - 1], filters[i], kernel_size, stride, padding, padding_mode, n_dims, conv_type
                )
            )
            layers.append(activation())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)


class EncoderBlock(AbstractEncoderBlock):
    def __init__(
        self,
        in_channels: int,
        filters: list,
        use_pooling: bool = True,
        n_dims: int = 1,
        conv_block_kwargs: dict = None,
        pooling_kwargs: dict = None,
    ):
        self.in_channels = in_channels
        self.out_channels = filters[-1]
        self.filters = filters

        self.use_pooling = use_pooling
        self.n_dims = n_dims

        conv_block_kwargs = self.__override_conv_block_kwargs(conv_block_kwargs)
        conv_block = ConvBlock(**conv_block_kwargs)
        pool = None
        if use_pooling:
            pooling_kwargs = self.__override_pooling_kwargs(pooling_kwargs)
            pool = get_pooling_layer(**pooling_kwargs)
        super(EncoderBlock, self).__init__(block=conv_block, pool=pool)

    def __override_conv_block_kwargs(self, conv_block_kwargs) -> dict:
        default_kwargs = {
            "in_channels": self.in_channels,
            "filters": self.filters,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "conv_type": "periodic",
            "activation": nn.ReLU,
            "n_dims": self.n_dims,
        }
        return self.override_kwargs(default_kwargs, conv_block_kwargs)

    def __override_pooling_kwargs(self, pooling_kwargs) -> dict:
        default_kwargs = {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "pool_type": "max",
            "n_dims": self.n_dims,
        }
        return self.override_kwargs(default_kwargs, pooling_kwargs)


class DecoderBlock(AbstractDecoderBlock):
    def __init__(
        self,
        in_channels: int,
        filters: list,
        n_dims: int = 1,
        use_upconv: bool = True,
        conv_block_kwargs: dict = None,
        upconv_kwargs: dict = None,
    ):
        self.in_channels = in_channels
        self.out_channels = filters[-1]
        self.filters = filters
        self.n_dims = n_dims
        if use_upconv:
            # self.filters[0] *= 2
            self.in_channels *= 2
        else:
            self.filters[0] = self.in_channels

        conv_block_kwargs = self.__override_conv_block_kwargs(conv_block_kwargs)
        conv_block = ConvBlock(**conv_block_kwargs)
        upconv = None
        if use_upconv:
            upconv_kwargs = self.__override_upconv_kwargs(upconv_kwargs)
            upconv = get_upconv_layer(**upconv_kwargs)
        super(DecoderBlock, self).__init__(block=conv_block, upscale=upconv)

    def __override_conv_block_kwargs(self, conv_block_kwargs) -> dict:
        default_kwargs = {
            "in_channels": self.in_channels,
            "filters": self.filters,
            "kernel_size": 3,
            "stride": 1,
            "padding": 3,
            "conv_type": "periodic",
            "activation": nn.ReLU,
            "n_dims": self.n_dims,
        }
        return self.override_kwargs(default_kwargs, conv_block_kwargs)

    def __override_upconv_kwargs(self, upconv_kwargs) -> dict:
        default_kwargs = {
            "in_channels": self.filters[0],
            "out_channels": self.filters[0],
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "n_dims": self.n_dims,
        }
        return self.override_kwargs(default_kwargs, upconv_kwargs)


class OutputBlock(AbstractOutputBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_dims: int = 1,
        conv_kwargs: dict = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dims = n_dims

        conv_kwargs = self.__override_conv_kwargs(conv_kwargs)
        conv = get_conv_layer(**conv_kwargs)
        super(OutputBlock, self).__init__(block=conv)

    def __override_conv_kwargs(self, conv_kwargs) -> dict:
        default_kwargs = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": 3,
            "padding_mode": "circular",
            "n_dims": self.n_dims,
            "conv_type": "periodic",
        }
        return self.override_kwargs(default_kwargs, conv_kwargs)
