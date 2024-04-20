# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import config
import torch
from typing import Union, Optional, Tuple
from diffusers import UNetSpatioTemporalConditionModel
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from dataclasses import dataclass

# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label



# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------

def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "sample": torch.rand((batchsize, 25, 8, 72, 128), dtype=torch_dtype),
        "timestep": torch.rand((1,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize , 1,  1024), dtype=torch_dtype),
        "added_time_ids": torch.rand((batchsize, 3), dtype=torch_dtype)
    }
    return inputs


def unet_load(model_name):
    model = UNetSpatioTemporalConditionModel.from_pretrained(model_name, subfolder="unet")
    return model


def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)
