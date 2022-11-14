import torch
import torch.nn as nn

from unet.layers import get_conv_layer, get_pooling_layer, get_upconv_layer


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


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_pooling: bool = True,
        n_dims: int = 1,
        conv_block_kwargs: dict = None,
        pooling_kwargs: dict = None,
    ):
        super(EncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_pooling = use_pooling
        if conv_block_kwargs is None:
            conv_block_kwargs = {
                "in_channels": in_channels,
                "filters": [out_channels, out_channels],
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "conv_type": "periodic",
                "activation": nn.ReLU,
                "n_dims": n_dims,
            }
        if pooling_kwargs is None:
            pooling_kwargs = {
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "pool_type": "max",
                "n_dims": n_dims,
            }
        self.conv_block = ConvBlock(**conv_block_kwargs)
        if use_pooling:
            self.pool = get_pooling_layer(**pooling_kwargs)
        # value of x before pooling
        self._x = None

    def forward(self, x):
        x = self.conv_block.forward(x)
        self._x = x
        if self.use_pooling:
            x = self.pool.forward(x)
        return x

    @property
    def x(self):
        return self._x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_upconv: bool = True,
        n_dims: int = 1,
        conv_block_kwargs: dict = None,
        upconv_kwargs: dict = None,
    ):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_upconv = use_upconv
        if conv_block_kwargs is None:
            conv_block_kwargs = {
                "in_channels": in_channels * 2 if use_upconv else in_channels,
                "filters": [out_channels * 2, out_channels] if use_upconv else [in_channels, out_channels],
                "kernel_size": 3,
                "stride": 1,
                "padding": 3,
                "conv_type": "periodic",
                "activation": nn.ReLU,
                "n_dims": n_dims,
            }
        if upconv_kwargs is None:
            upconv_kwargs = {
                "in_channels": in_channels,
                "out_channels": in_channels,
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "n_dims": n_dims,
            }
        self.conv_block = ConvBlock(**conv_block_kwargs)
        if use_upconv:
            self.upconv = get_upconv_layer(**upconv_kwargs)

    def forward(self, x, x_skip=None):
        if self.use_upconv:
            x = self.upconv.forward(x)
            x = torch.concat((x, x_skip), dim=1)
        x = self.conv_block.forward(x)
        return x


class OutputConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_dims: int,
        conv_type: str = "default",
        conv_block_kwargs: dict = None,
    ):
        super(OutputConv, self).__init__()
        if conv_block_kwargs is None:
            conv_block_kwargs = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "padding_mode": "zeros",
                "n_dims": n_dims,
                "conv_type": conv_type,
            }
        self.conv = get_conv_layer(**conv_block_kwargs)

    def forward(self, x):
        return self.conv.forward(x)
