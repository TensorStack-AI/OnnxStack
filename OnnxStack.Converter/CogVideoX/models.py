import config
import torch
from typing import Union, Tuple
from diffusers import CogVideoXTransformer3DModel,AutoencoderKLCogVideoX


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
# VAE DECODER
# -----------------------------------------------------------------------------
def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((batchsize, 16, 16, 60, 90), dtype=torch_dtype)
    }


def vae_decoder_load(model_name):
    model = AutoencoderKLCogVideoX.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float32)




# -----------------------------------------------------------------------------
# TRANSFORMER
# -----------------------------------------------------------------------------
class WrappedCogVideoXTransformer3DModel(CogVideoXTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.FloatTensor, 
        encoder_hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor,
    ) -> Union[CogVideoXTransformer3DModel, Tuple]:

        return super().forward(
            hidden_states= hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep = timestep,
            timestep_cond = None
        )


def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "hidden_states": torch.rand((1, 13, 16, 60, 90), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((1, 226, 4096), dtype=torch_dtype),
        "timestep": torch.rand((1), dtype=torch_dtype)
    }
    return inputs


def unet_load(model_name):
    model = CogVideoXTransformer3DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.float32)
    return model


def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)