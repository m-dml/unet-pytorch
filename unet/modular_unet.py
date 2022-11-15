from typing import Union

import torch.nn as nn

from unet.blocks import AbstractDecoderBlock, AbstractEncoderBlock, AbstractOutputBlock


class ModularUnet(nn.Module):
    def __init__(
        self,
        encoder_blocks: Union[nn.ModuleList, list, tuple],
        decoder_blocks: Union[nn.ModuleList, list, tuple],
        output_block: Union[nn.Module],
        skip_connection: nn.Module,
        use_bottom_skip: bool = False,
    ):
        super(ModularUnet, self).__init__()
        encoder_size = len(encoder_blocks)
        decoder_size = len(decoder_blocks)
        if encoder_size != decoder_size:
            raise ValueError("encoder_blocks abd decoder_blocks length mismatch ")
        self.n_levels = encoder_size

        for block in encoder_blocks:
            if not isinstance(block, AbstractEncoderBlock):
                raise TypeError("blocks in 'encoder_blocks' must be AbstractEncoderBlock or inherited from it")
        self.encoder_blocks = encoder_blocks

        for block in decoder_blocks:
            if not isinstance(block, AbstractDecoderBlock):
                raise TypeError("blocks in 'decoder_blocks' must contain AbstractDecoderBlock or inherited from it")
        self.decoder_blocks = decoder_blocks

        if not isinstance(output_block, AbstractOutputBlock):
            raise TypeError("output_block must be an AbstractOutputBlock or inherited from it")
        self.output_block = output_block

        self.skip_connection = skip_connection
        self.use_bottom_skip = use_bottom_skip

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block.forward(x)

        if self.use_bottom_skip:
            x = self.skip_connection(x)
        x = self.decoder_blocks[0].forward(x)
        for i in range(1, self.n_levels):
            j = self.n_levels - 1 - i
            x_skip = self.skip_connection(self.encoder_blocks[j].x)
            x = self.decoder_blocks[i].forward(x, x_skip)

        x = self.output_block.forward(x)
        return x
