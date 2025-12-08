import torch
from typing import Union, Tuple
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from transformers import  UMT5EncoderModel


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
        "input_ids": torch.zeros((batchsize, 512), dtype=torch_dtype)
    }


def text_encoder_load(model_name):
    return UMT5EncoderModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=torch.float32)


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int64)


def text_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int64)



# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------
def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((1, 16, 21, 60, 104), dtype=torch_dtype)
    }


def vae_decoder_load(model_name):
    model = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)



# -----------------------------------------------------------------------------
# TRANSFORMER
# -----------------------------------------------------------------------------
class WrappedWanTransformer3DModel(WanTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.Tensor, 
        timestep: torch.LongTensor, 
        encoder_hidden_states: torch.Tensor
    ) -> Union[WanTransformer3DModel, Tuple]:
        return super().forward(
            hidden_states = hidden_states,
            timestep = timestep,
            encoder_hidden_states = encoder_hidden_states
        )


def transformer_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "hidden_states": torch.rand((1, 16, 21, 60, 104), dtype=torch_dtype),
        "timestep": torch.rand((1,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((1, 512, 4096), dtype=torch_dtype)
    }
    return inputs


def transformer_load(model_name):
    model = WrappedWanTransformer3DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.float32)
    return model


def transformer_conversion_inputs(model=None):
    return tuple(transformer_inputs(1, torch.float32, True).values())


def transformer_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(transformer_inputs, batchsize, torch.float16)