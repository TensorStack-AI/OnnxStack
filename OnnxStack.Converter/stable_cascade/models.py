# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import config
import torch
from typing import Union, Optional, Tuple
from diffusers import AutoencoderKL, StableCascadeUNet
from diffusers.pipelines.wuerstchen import PaellaVQModel
from transformers.models.clip.modeling_clip import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
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
# TEXT ENCODER
# -----------------------------------------------------------------------------

def text_encoder_inputs(batchsize, torch_dtype):
    return torch.zeros((batchsize, 77), dtype=torch_dtype)


def text_encoder_load(model_name):
    model = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder")
    return model


def text_encoder_conversion_inputs(model=None):
    return text_encoder_inputs(1, torch.int32)


def text_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int32)




# -----------------------------------------------------------------------------
# DECODER UNET
# -----------------------------------------------------------------------------

def decoder_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    # TODO(jstoecker): Rename onnx::Concat_4 to text_embeds and onnx::Shape_5 to time_ids
    inputs = {
        "sample": torch.rand((batchsize, 4, 256, 256), dtype=torch_dtype),
        "timestep_ratio": torch.rand((batchsize,), dtype=torch_dtype),
        "clip_text_pooled": torch.rand((batchsize , 1,  1280), dtype=torch_dtype),
        "effnet": torch.rand((batchsize, 16, 24, 24), dtype=torch_dtype)
    }

    # use as kwargs since they won't be in the correct position if passed along with the tuple of inputs
    kwargs = {
        "return_dict": False,
    }
   
    return inputs


def decoder_load(model_name):
    model = StableCascadeUNet.from_pretrained(model_name, subfolder="decoder")
    return model


def decoder_conversion_inputs(model=None):
    return tuple(decoder_inputs(1, torch.float32, True).values())


def decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(decoder_inputs, batchsize, torch.float16)




# -----------------------------------------------------------------------------
# PRIOR UNET
# -----------------------------------------------------------------------------

def prior_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "sample": torch.rand((batchsize, 16, 24, 24), dtype=torch_dtype),
        "timestep_ratio": torch.rand((batchsize,), dtype=torch_dtype),
        "clip_text_pooled": torch.rand((batchsize  , 1,  1280), dtype=torch_dtype),
        "clip_text": torch.rand((batchsize  , 77,  1280), dtype=torch_dtype),
        "clip_img": torch.rand((batchsize , 1,  768), dtype=torch_dtype)
    }

    # use as kwargs since they won't be in the correct position if passed along with the tuple of inputs
    kwargs = {
        "return_dict": False,
    }
   
    return inputs


def prior_load(model_name):
    model = StableCascadeUNet.from_pretrained(model_name, subfolder="prior")
    return model


def prior_conversion_inputs(model=None):
    return tuple(prior_inputs(1, torch.float32, True).values())


def prior_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(prior_inputs, batchsize, torch.float16)



    
# -----------------------------------------------------------------------------
# IMAGE ENCODER
# -----------------------------------------------------------------------------

def image_encoder_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "sample": torch.rand((batchsize, 3, 224, 224), dtype=torch_dtype)
    }
    return inputs


def image_encoder_load(model_name):
    model = CLIPVisionModelWithProjection.from_pretrained(model_name, subfolder="image_encoder", use_safetensors=True)
    return model


def image_encoder_conversion_inputs(model=None):
    return tuple(image_encoder_inputs(1, torch.float32, True).values())


def image_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(image_encoder_inputs, batchsize, torch.float16)




# -----------------------------------------------------------------------------
# VQGAN
# -----------------------------------------------------------------------------

def vqgan_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "sample": torch.rand((batchsize, 3, 256, 256), dtype=torch_dtype)
    }
    return inputs


def vqgan_load(model_name):
    model = PaellaVQModel.from_pretrained(model_name, subfolder="vqgan", use_safetensors=True)
    return model


def vqgan_conversion_inputs(model=None):
    return tuple(vqgan_inputs(1, torch.float32, True).values())


def vqgan_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vqgan_inputs, batchsize, torch.float16)