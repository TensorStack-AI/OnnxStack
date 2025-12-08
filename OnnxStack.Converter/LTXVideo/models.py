import torch
from typing import Union, Tuple
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
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
# VAE DECODER
# -----------------------------------------------------------------------------
def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((batchsize, 128, 21, 15, 22), dtype=torch_dtype)
    }


def vae_decoder_load(model_name):
    model = AutoencoderKLLTXVideo.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# TRANSFORMER
# -----------------------------------------------------------------------------
class WrappedLTXVideoTransformer3DModel(LTXVideoTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.FloatTensor, 
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        timestep: torch.FloatTensor, 
        num_frames: int, 
        height: int, 
        width: int, 
        rope_interpolation_scale: torch.FloatTensor, 
    ) -> Union[LTXVideoTransformer3DModel, Tuple]:

        return super().forward(
            hidden_states= hidden_states,
            encoder_hidden_states= encoder_hidden_states,
            timestep= timestep,
            encoder_attention_mask= encoder_attention_mask,
            num_frames= num_frames,
            height= height,
            width= width,
            rope_interpolation_scale= rope_interpolation_scale,
            video_coords= None,
            attention_kwargs= None,
            return_dict = True
        )


def transformer_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "hidden_states": torch.rand((batchsize, 6930, 128), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 128, 4096), dtype=torch_dtype),
        "encoder_attention_mask": torch.rand((batchsize, 128), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "num_frames": 21,
        "height": 15,
        "width": 22,
        "rope_interpolation_scale": torch.tensor([0.32, 32, 32], dtype=torch_dtype)
    }
    return inputs


def transformer_load(model_name):
    model = WrappedLTXVideoTransformer3DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.float32)
    return model


def transformer_conversion_inputs(model=None):
    return tuple(transformer_inputs(1, torch.float32, True).values())


def transformer_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(transformer_inputs, batchsize, torch.float16)


