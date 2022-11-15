import torch.nn as nn

from unet.blocks import DecoderBlock, EncoderBlock, OutputConv
from unet.layers import get_global_pooling_layer


class AsymmetricUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: int = 3,
        encoder_dim: int = 2,
        decoder_dim: int = 1,
        conv_type: str = "default",
        global_pooling_type: str = "max",
    ):
        super(AsymmetricUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_levels = n_levels
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.conv_type = conv_type
        self.encoder = self.instantiate_encoder()
        self.decoder = self.instantiate_decoder()
        self.output = OutputConv(
            in_channels=self.encoder[0].out_channels,
            out_channels=out_channels,
            n_dims=decoder_dim,
            conv_kwargs={
                "conv_type": self.conv_type,
            },
        )
        self.gp = get_global_pooling_layer(type=global_pooling_type)

    def instantiate_encoder(self):
        filters = [32]
        for i in range(1, self.n_levels):
            filters.append(filters[-1] * 2)

        encoder = nn.ModuleList()
        encoder.append(
            EncoderBlock(
                in_channels=self.in_channels,
                out_channels=filters[0],
                use_pooling=True,
                n_dims=self.encoder_dim,
                conv_block_kwargs={
                    "conv_type": self.conv_type,
                },
            )
        )
        for i in range(1, self.n_levels):
            use_pooling = True
            if i == self.n_levels - 1:
                use_pooling = False
            encoder.append(
                EncoderBlock(
                    in_channels=filters[i - 1],
                    out_channels=filters[i],
                    use_pooling=use_pooling,
                    n_dims=self.encoder_dim,
                    conv_block_kwargs={
                        "conv_type": self.conv_type,
                    },
                )
            )
        return encoder

    def instantiate_decoder(self):
        decoder = nn.ModuleList()
        for i in range(self.n_levels - 1, -1, -1):
            use_upconv = True
            if i == self.n_levels - 1:
                use_upconv = False
            out_channels = self.encoder[i].in_channels
            if i == 0:
                out_channels = self.encoder[0].out_channels
            decoder.append(
                DecoderBlock(
                    in_channels=self.encoder[i].out_channels,
                    out_channels=out_channels,
                    use_upconv=use_upconv,
                    n_dims=self.decoder_dim,
                    conv_block_kwargs={
                        "conv_type": self.conv_type,
                    },
                )
            )
        return decoder

    def forward(self, x):
        for block in self.encoder:
            x = block.forward(x)

        x = self.decoder[0].forward(self.gp(x))
        for i in range(1, self.n_levels):
            j = self.n_levels - 1 - i
            x = self.decoder[i].forward(x, self.gp(self.encoder[j].x))

        x = self.output.forward(x)
        return x
