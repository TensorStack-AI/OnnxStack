import config
import torch
from typing import Union, Tuple
from diffusers import AutoencoderKL, PixArtTransformer2DModel
from transformers import T5EncoderModel

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
    return {
        "input_ids": torch.zeros((batchsize, config.text_max_sequence), dtype=torch_dtype)
    }


def text_encoder_load(model_name):
    return T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder")


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int64)


def text_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int64)




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
        "latent_sample": torch.rand((batchsize, config.unet_channels, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype)
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
# TRANSFORMER
# -----------------------------------------------------------------------------

class WrappedPixArtTransformer2DModel(PixArtTransformer2DModel):
    def forward(
        self,
        hidden_states: torch.FloatTensor, 
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        timestep: torch.LongTensor,
    ) -> Union[PixArtTransformer2DModel, Tuple]:
        return super().forward(
            hidden_states = hidden_states,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = encoder_attention_mask,
            timestep = timestep
        )


def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "hidden_states": torch.rand((batchsize, config.unet_channels, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, config.text_max_sequence, config.text_length), dtype=torch_dtype),
        "encoder_attention_mask": torch.rand((batchsize, config.text_max_sequence), dtype=torch_dtype),
        "timestep": torch.rand((batchsize), dtype=torch_dtype)
    }
    return inputs


def unet_load(model_name):
    model = WrappedPixArtTransformer2DModel.from_pretrained(model_name, subfolder="transformer")
    return model


def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)