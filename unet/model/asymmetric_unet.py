import torch.nn as nn

from unet.layers import get_global_pooling_layer
from unet.model.blocks import DecoderBlock, EncoderBlock, OutputBlock
from unet.model.modular_unet import ModularUnet


class Unet(ModularUnet):
    def __init__(
        self,
        in_channels: int,
        encoder_layers: list,
        decoder_layers: list,
        out_channels: int,
        encoder_dim: int = 2,
        decoder_dim: int = 1,
        conv_type: str = "default",
        global_pooling_type: str = "max",
    ):
        encoder_size = len(encoder_layers)
        decoder_size = len(decoder_layers)
        if encoder_size != decoder_size:
            raise ValueError(
                f"number of encoder layers ({encoder_size}) and decoder layers ({decoder_size}) do not match"
            )
        self.n_levels = encoder_size

        encoder_blocks = self.instantiate_encoder(in_channels, encoder_layers, conv_type, encoder_dim)
        decoder_blocks = self.instantiate_decoder(
            encoder_blocks[-1].out_channels, decoder_layers, conv_type, decoder_dim
        )
        output_block = self.instantiate_output_block(
            decoder_blocks[-1].out_channels, out_channels, conv_type, decoder_dim
        )
        skip_connection = get_global_pooling_layer(type=global_pooling_type)

        super(Unet, self).__init__(
            encoder_blocks=encoder_blocks,
            decoder_blocks=decoder_blocks,
            output_block=output_block,
            skip_connection=skip_connection,
            use_bottom_skip=True,
        )

    def instantiate_encoder(self, in_channels: int, encoder_layers: list, conv_type: str, encoder_dim: int):
        encoder = nn.ModuleList()
        for i in range(self.n_levels):
            if i > 0:
                in_channels = encoder_layers[i - 1][-1]
            use_pooling = True
            if i == self.n_levels - 1:
                use_pooling = False
            block = EncoderBlock(
                in_channels=in_channels,
                filters=encoder_layers[i],
                use_pooling=use_pooling,
                n_dims=encoder_dim,
                conv_block_kwargs={
                    "conv_type": conv_type,
                },
            )
            encoder.append(block)
        return encoder

    def instantiate_decoder(self, in_channels: int, decoder_layers: list, conv_type: str, decoder_dim: int):
        decoder = nn.ModuleList()
        for i in range(self.n_levels):
            use_upconv = False
            if i > 0:
                use_upconv = True
                in_channels = decoder_layers[i - 1][-1]
            block = DecoderBlock(
                in_channels=in_channels,
                filters=decoder_layers[i],
                use_upconv=use_upconv,
                n_dims=decoder_dim,
                conv_block_kwargs={
                    "conv_type": conv_type,
                },
            )
            decoder.append(block)
        return decoder

    def instantiate_output_block(self, in_channels: int, out_channels: int, conv_type: str, decoder_dim: int):
        return OutputBlock(
            in_channels,
            out_channels,
            n_dims=decoder_dim,
            conv_kwargs={
                "conv_type": conv_type,
            },
        )
