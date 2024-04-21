# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import config
import torch
from typing import Union, Tuple
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection

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
    model = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    return model


def text_encoder_conversion_inputs(model=None):
    return text_encoder_inputs(1, torch.int32)


def text_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int32)




# -----------------------------------------------------------------------------
# TEXT ENCODER 2
# -----------------------------------------------------------------------------

def text_encoder_2_inputs(batchsize, torch_dtype):
    return {
        "input_ids": torch.zeros((batchsize, 77), dtype=torch_dtype),
        "output_hidden_states": True,
    }


def text_encoder_2_load(model_name):
    return CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2")


def text_encoder_2_conversion_inputs(model):
    return text_encoder_2_inputs(1, torch.int64)


def text_encoder_2_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_2_inputs, batchsize, torch.int64)




# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------

class SDXLUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor, 
        timestep: torch.FloatTensor, 
        encoder_hidden_states: torch.FloatTensor,
        text_embeds: torch.FloatTensor,
        time_ids: torch.FloatTensor
    ) -> Union[UNet2DConditionModel, Tuple]:
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        return super().forward(
            sample = sample,
            timestep = timestep,
            encoder_hidden_states = encoder_hidden_states,
            added_cond_kwargs = added_cond_kwargs
        )


def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "sample": torch.rand((batchsize, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, config.cross_attention_dim), dtype=torch_dtype),
        "text_embeds": torch.rand((1, config.text_embeds_size), dtype=torch_dtype),
        "time_ids": torch.rand((1, config.time_ids_size), dtype=torch_dtype),
    }
    return inputs


def unet_load(model_name):
    model = SDXLUNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    return model


def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)




# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------

def vae_encoder_inputs(batchsize, torch_dtype):
    return {"sample": torch.rand((batchsize, 3, config.vae_sample_size, config.vae_sample_size), dtype=torch_dtype)}


def vae_encoder_load(model_name):
    model = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
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
        "latent_sample": torch.rand((batchsize, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype)
    }


def vae_decoder_load(model_name):
    model = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)




# -----------------------------------------------------------------------------
# CONTROLNET - UNET
# -----------------------------------------------------------------------------

class ControlNetUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.FloatTensor,
        time_ids: torch.FloatTensor,
        down_block_0_additional_residual: torch.Tensor,
        down_block_1_additional_residual: torch.Tensor,
        down_block_2_additional_residual: torch.Tensor,
        down_block_3_additional_residual: torch.Tensor,
        down_block_4_additional_residual: torch.Tensor,
        down_block_5_additional_residual: torch.Tensor,
        down_block_6_additional_residual: torch.Tensor,
        down_block_7_additional_residual: torch.Tensor,
        down_block_8_additional_residual: torch.Tensor,
        mid_block_additional_residual: torch.Tensor,
    ) -> Union[UNet2DConditionModel, Tuple]:
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        down_block_add_res = (
            down_block_0_additional_residual, down_block_1_additional_residual, down_block_2_additional_residual,
            down_block_3_additional_residual, down_block_4_additional_residual, down_block_5_additional_residual,
            down_block_6_additional_residual, down_block_7_additional_residual, down_block_8_additional_residual)
        return super().forward(
            sample = sample,
            timestep = timestep,
            encoder_hidden_states = encoder_hidden_states,
            added_cond_kwargs = added_cond_kwargs,
            down_block_additional_residuals = down_block_add_res,
            mid_block_additional_residual = mid_block_additional_residual,
            return_dict = False
        )

def controlnet_unet_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, config.cross_attention_dim), dtype=torch_dtype),
        "text_embeds": torch.rand(( batchsize, 1280), dtype=torch_dtype),
        "time_ids": torch.rand((batchsize, config.time_ids_size), dtype=torch_dtype),
        "down_block_0_additional_residual": torch.rand((batchsize, 320, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "down_block_1_additional_residual": torch.rand((batchsize, 320, config.unet_sample_size , config.unet_sample_size), dtype=torch_dtype),
        "down_block_2_additional_residual": torch.rand((batchsize, 320, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "down_block_3_additional_residual": torch.rand((batchsize, 320, config.unet_sample_size // 2, config.unet_sample_size // 2), dtype=torch_dtype),
        "down_block_4_additional_residual": torch.rand((batchsize, 640, config.unet_sample_size // 2, config.unet_sample_size // 2), dtype=torch_dtype),
        "down_block_5_additional_residual": torch.rand((batchsize, 640, config.unet_sample_size // 2, config.unet_sample_size // 2), dtype=torch_dtype),
        "down_block_6_additional_residual": torch.rand((batchsize, 640, config.unet_sample_size // 4, config.unet_sample_size // 4), dtype=torch_dtype),
        "down_block_7_additional_residual": torch.rand((batchsize, 1280, config.unet_sample_size // 4, config.unet_sample_size // 4), dtype=torch_dtype),
        "down_block_8_additional_residual": torch.rand((batchsize, 1280, config.unet_sample_size // 4, config.unet_sample_size // 4), dtype=torch_dtype),
        "mid_block_additional_residual": torch.rand((batchsize, 1280, config.unet_sample_size // 4, config.unet_sample_size // 4), dtype=torch_dtype)
    }


def controlnet_unet_load(model_name):
    model = ControlNetUNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    return model


def controlnet_unet_conversion_inputs(model):
    return tuple(controlnet_unet_inputs(1, torch.float32).values())


def controlnet_unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(controlnet_unet_inputs, batchsize, torch.float16)