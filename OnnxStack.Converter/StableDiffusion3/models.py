import config
import torch
from typing import Union, Tuple
from diffusers import AutoencoderKL, SD3Transformer2DModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel


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
    model = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=torch.float32)
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
    return CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=torch.float32)


def text_encoder_2_conversion_inputs(model):
    return text_encoder_2_inputs(1, torch.int64)


def text_encoder_2_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_2_inputs, batchsize, torch.int64)



# -----------------------------------------------------------------------------
# TEXT ENCODER 3
# -----------------------------------------------------------------------------
def text_encoder_3_inputs(batchsize, torch_dtype):
    return {
        "input_ids": torch.zeros((batchsize, 512), dtype=torch_dtype)
    }


def text_encoder_3_load(model_name):
    return T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_3", torch_dtype=torch.float32)


def text_encoder_3_conversion_inputs(model):
    return text_encoder_3_inputs(1, torch.int64)


def text_encoder_3_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_3_inputs, batchsize, torch.int64)



# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------
def vae_encoder_inputs(batchsize, torch_dtype):
    return {"sample": torch.rand((batchsize, 3, config.vae_sample_size, config.vae_sample_size), dtype=torch_dtype)}


def vae_encoder_load(model_name):
    model = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
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
        "latent_sample": torch.rand((batchsize, 16, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype)
    }


def vae_decoder_load(model_name):
    model = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)



# -----------------------------------------------------------------------------
# TRANSFORMER
# -----------------------------------------------------------------------------
class WrappedSD3Transformer2DModel(SD3Transformer2DModel):
    def forward(
        self,
        hidden_states: torch.FloatTensor, 
        timestep: torch.FloatTensor, 
        encoder_hidden_states: torch.FloatTensor,
        pooled_projections: torch.FloatTensor
    ) -> Union[SD3Transformer2DModel, Tuple]:

        return super().forward(
            hidden_states = hidden_states,
            encoder_hidden_states = encoder_hidden_states,
            pooled_projections = pooled_projections,
            timestep = timestep,
            block_controlnet_hidden_states = None,
            joint_attention_kwargs = None
        )


def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "hidden_states": torch.rand((batchsize, 16, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, 4096), dtype=torch_dtype),
        "pooled_projections": torch.rand((1, 2048), dtype=torch_dtype)
    }
    return inputs


def unet_load(model_name):
    model = WrappedSD3Transformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.float32)
    return model


def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)



# -----------------------------------------------------------------------------
# CONTROLNET - TRANSFORMER
# -----------------------------------------------------------------------------

class WrappedSD3Transformer2DControlNetModel(SD3Transformer2DModel):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep:torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        pooled_projections: torch.FloatTensor,
        controlnet_block_sample: torch.FloatTensor,
    ) -> Union[SD3Transformer2DModel, Tuple]:
        return super().forward(
            hidden_states = hidden_states,
            encoder_hidden_states = encoder_hidden_states,
            pooled_projections = pooled_projections,
            timestep = timestep,
            block_controlnet_hidden_states = controlnet_block_sample,
            return_dict = False
        )


def controlnet_unet_inputs(batchsize, torch_dtype):
    return {
        "hidden_states": torch.rand((batchsize, 16, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 333, 4096), dtype=torch_dtype),
        "pooled_projections": torch.rand((1, 2048), dtype=torch_dtype),
        "controlnet_block_sample": torch.rand((12, 4096, 1536), dtype=torch_dtype)
    }


def controlnet_unet_load(model_name):
    model = WrappedSD3Transformer2DControlNetModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.float32)
    return model


def controlnet_unet_conversion_inputs(model):
    return tuple(controlnet_unet_inputs(1, torch.float32).values())


def controlnet_unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(controlnet_unet_inputs, batchsize, torch.float16)