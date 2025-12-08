import config
import torch
from typing import Union, Optional, Tuple
from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
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
        "sample": torch.rand((1, 14, 8, 72, 128), dtype=torch_dtype),
        "timestep": torch.rand((1), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((1 , 1,  1024), dtype=torch_dtype),
        "added_time_ids": torch.rand((1, 3), dtype=torch_dtype)
    }
    return inputs


def unet_load(model_name):
    model = UNetSpatioTemporalConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=torch.float32)
    return model


def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)



# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------
def vae_encoder_inputs(batchsize, torch_dtype):
    return {"sample": torch.rand((batchsize, 3, 72, 128), dtype=torch_dtype)}


def vae_encoder_load(model_name):
    model = AutoencoderKLTemporalDecoder.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    model.forward = lambda sample: model.encode(sample)[0].sample()
    return model


def vae_encoder_conversion_inputs(model=None):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def vae_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, batchsize, torch.float16)



# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------
def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((14, 4, 72, 128), dtype=torch_dtype),
        "num_frames": 1,
    }


def vae_decoder_load(model_name):
    model = AutoencoderKLTemporalDecoder.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)