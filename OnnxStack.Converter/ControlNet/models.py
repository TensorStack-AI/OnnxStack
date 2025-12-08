import torch
import numpy as np
from typing import Union, Tuple
from dataclasses import dataclass
from diffusers import ControlNetModel
from diffusers.utils import BaseOutput
from diffusers.models.controlnet import ControlNetOutput 


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
    down_block_0_additional_residual: torch.Tensor
    down_block_1_additional_residual: torch.Tensor
    down_block_2_additional_residual: torch.Tensor
    down_block_3_additional_residual: torch.Tensor
    down_block_4_additional_residual: torch.Tensor
    down_block_5_additional_residual: torch.Tensor
    down_block_6_additional_residual: torch.Tensor
    down_block_7_additional_residual: torch.Tensor
    down_block_8_additional_residual: torch.Tensor
    down_block_9_additional_residual: torch.Tensor
    down_block_10_additional_residual: torch.Tensor
    down_block_11_additional_residual: torch.Tensor
    mid_block_additional_residual: torch.Tensor


class PatchedControlNetModel(ControlNetModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float,
    ) -> Union[ControlNetOutput, Tuple]:
        (down_block_res_samples, mid_block_res_sample) = super().forward(
            sample = sample,
            timestep = timestep,
            encoder_hidden_states = encoder_hidden_states,
            controlnet_cond = controlnet_cond,
            conditioning_scale = conditioning_scale,
            return_dict = False
        )

        return PatchedControlNetOutput(
            down_block_0_additional_residual = down_block_res_samples[0],
            down_block_1_additional_residual = down_block_res_samples[1],
            down_block_2_additional_residual = down_block_res_samples[2],
            down_block_3_additional_residual = down_block_res_samples[3],
            down_block_4_additional_residual = down_block_res_samples[4],
            down_block_5_additional_residual = down_block_res_samples[5],
            down_block_6_additional_residual = down_block_res_samples[6],
            down_block_7_additional_residual = down_block_res_samples[7],
            down_block_8_additional_residual = down_block_res_samples[8],
            down_block_9_additional_residual = down_block_res_samples[9],
            down_block_10_additional_residual = down_block_res_samples[10],
            down_block_11_additional_residual = down_block_res_samples[11],
            mid_block_additional_residual = mid_block_res_sample
        )


# -----------------------------------------------------------------------------
# ControlNet
# -----------------------------------------------------------------------------
def controlnet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    return {
        "sample": torch.rand((batchsize, 4, 64, 64), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, 768), dtype=torch_dtype),
        "controlnet_cond": torch.rand((batchsize, 3, 512, 512), dtype=torch_dtype),
        "conditioning_scale": 1.0,
    }


def controlnet_load(model_name):
    model = PatchedControlNetModel.from_pretrained(model_name)
    return model


def controlnet_conversion_inputs(model=None):
    return tuple(controlnet_inputs(1, torch.float32, True).values())


def controlnet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(controlnet_inputs, batchsize, torch.float32)