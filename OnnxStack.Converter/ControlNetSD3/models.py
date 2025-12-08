import torch
import numpy as np
from typing import Union, Tuple
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.models.controlnets import SD3ControlNetModel, SD3ControlNetOutput


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label


@dataclass
class PatchedControlNetOutput(BaseOutput):
    controlnet_block_sample: torch.FloatTensor


class PatchedControlNetModel(SD3ControlNetModel):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        pooled_projections: torch.FloatTensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float
    ) -> Union[SD3ControlNetOutput, Tuple]:
        controlnet_result = super().forward(
            hidden_states = hidden_states,
            controlnet_cond = controlnet_cond,
            conditioning_scale = conditioning_scale,
            encoder_hidden_states = encoder_hidden_states,
            pooled_projections = pooled_projections,
            timestep = timestep
        )

        controlnet_block_samples = torch.cat(list(controlnet_result[0]), dim=0)
        return PatchedControlNetOutput(
            controlnet_block_sample = controlnet_block_samples
        )


# -----------------------------------------------------------------------------
# ControlNet
# -----------------------------------------------------------------------------
def controlnet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    return {
        "hidden_states": torch.rand((batchsize, 16, 128, 128), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 333, 4096), dtype=torch_dtype),
        "pooled_projections": torch.rand((1, 2048), dtype=torch_dtype),
        "controlnet_cond": torch.rand((batchsize, 16, 128, 128), dtype=torch_dtype),
        "conditioning_scale": 1.0
    }


def controlnet_load(model_name):
    model = PatchedControlNetModel.from_pretrained(model_name)
    return model


def controlnet_conversion_inputs(model=None):
    return tuple(controlnet_inputs(1, torch.float32, True).values())


def controlnet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(controlnet_inputs, batchsize, torch.float32)