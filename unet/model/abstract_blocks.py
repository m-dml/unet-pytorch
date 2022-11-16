from typing import Union

import torch
import torch.nn as nn


class AbstractBlock(nn.Module):
    @staticmethod
    def override_kwargs(
        default_kwargs: dict,
        override_kwargs: dict,
    ) -> dict:
        default_keys = default_kwargs.keys()
        if override_kwargs is None:
            override_kwargs = {}
        for key, value in override_kwargs.items():
            if key not in default_keys:
                raise KeyError(f"{key} not allowed for dict with {default_keys}.")
            default_kwargs[key] = value
        return default_kwargs


class AbstractEncoderBlock(AbstractBlock):
    def __init__(
        self,
        block: Union[nn.Module, nn.Sequential] = None,
        pool: Union[nn.Module] = None,
    ):
        super(AbstractEncoderBlock, self).__init__()
        self.block = block
        self.pool = pool
        # value of x before pooling
        self._x = None

    @property
    def x(self):
        return self._x

    def forward(self, x):
        if self.block is None:
            raise ValueError("Provide a block in __init__ method")

        x = self.block.forward(x)
        self._x = x
        if self.pool is not None:
            x = self.pool.forward(x)
        return x


class AbstractDecoderBlock(AbstractBlock):
    def __init__(
        self,
        block: Union[nn.Module, nn.Sequential],
        upscale: Union[nn.Module] = None,
    ):
        super(AbstractDecoderBlock, self).__init__()
        self.block = block
        self.upscale = upscale

    def forward(self, x, x_skip=None):
        if self.block is None:
            raise ValueError("Provide a block in __init__ method")

        if self.upscale is not None:
            x = self.upscale.forward(x)
            x = torch.concat((x, x_skip), dim=1)
        x = self.block.forward(x)
        return x


class AbstractOutputBlock(AbstractBlock):
    def __init__(
        self,
        block: Union[nn.Module, nn.Sequential],
    ):
        super(AbstractOutputBlock, self).__init__()
        self.block = block

    def forward(self, x):
        if self.block is None:
            raise ValueError("Provide a block in __init__ method")
        return self.block.forward(x)
