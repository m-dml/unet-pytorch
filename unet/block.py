import torch
import torch.nn as nn

from unet.layers import get_conv_layer, get_pooling_layer


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


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        blocks: list,
        kernel_size: int,
        stride: int,
        padding: int,
        conv_type: str,
        n_dims: int,
        pool_type: str = "max",
        pool_size: int = 2,
    ):
        super(Encoder, self).__init__()
        self.blocks = self.instantiate_blocks(in_channels, blocks, kernel_size, stride, padding, conv_type, n_dims)
        self.pool = get_pooling_layer(kernel_size=pool_size, pool_type=pool_type, n_dims=n_dims)

    @staticmethod
    def instantiate_blocks(
        in_channels: int, blocks: list, kernel_size: int, stride: int, padding: int, conv_type: str, n_dims: int
    ) -> nn.ModuleList:
        encoder_blocks = nn.ModuleList()
        in_channels = in_channels
        for i, filters in enumerate(blocks):
            if i > 0:
                in_channels = blocks[i - 1][-1]
            encoder_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_type=conv_type,
                    n_dims=n_dims,
                )
            )
        return encoder_blocks

    def forward(self, x):
        output = []
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
            output.append(x)
            if i < len(self.blocks) - 1:
                x = self.pool.forward(x)
        return output
