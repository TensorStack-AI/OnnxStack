import torch
from typing import Union, Tuple
from diffusers import AutoencoderKL, ChromaTransformer2DModel
from transformers import T5EncoderModel

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
    return T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=torch.float32)


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int64)


def text_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int64)



# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------
def vae_encoder_inputs(batchsize, torch_dtype):
    return {"sample": torch.rand((batchsize, 3, 1024, 1024), dtype=torch_dtype)}


def vae_encoder_load(model_name):
    model = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    model.forward = lambda sample: model.encode(sample)[0].sample()
    return model


def vae_encoder_conversion_inputs(model=None):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def vae_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, batchsize, torch.float32)



# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------
def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((batchsize, 16, 128, 128), dtype=torch_dtype)
    }


def vae_decoder_load(model_name):
    model = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float32)



# -----------------------------------------------------------------------------
# TRANSFORMER
# -----------------------------------------------------------------------------
class WrappedChromaTransformer2DModel(ChromaTransformer2DModel):
    def forward(
        self,
        hidden_states: torch.FloatTensor, 
        encoder_hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor, 
        img_ids: torch.FloatTensor,
        txt_ids: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
    ) -> Union[ChromaTransformer2DModel, Tuple]:
        return super().forward(
            hidden_states = hidden_states,
            encoder_hidden_states = encoder_hidden_states,
            timestep = timestep,
            img_ids = img_ids,
            txt_ids = txt_ids,
            attention_mask=attention_mask,
            joint_attention_kwargs = None
        )


def transformer_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "hidden_states": torch.rand((1, 4096, 64), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((1, 512, 4096), dtype=torch_dtype),
        "timestep": torch.rand((1), dtype=torch_dtype),
        "img_ids": torch.rand((4096, 3), dtype=torch_dtype),
        "txt_ids": torch.rand((512, 3), dtype=torch_dtype),
        "attention_mask": torch.rand((1, 4608), dtype=torch_dtype)
    }
    return inputs


def transformer_load(model_name):
    model = WrappedChromaTransformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.float32)
    return model


def transformer_conversion_inputs(model=None):
    return tuple(transformer_inputs(1, torch.float32, True).values())


def transformer_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(transformer_inputs, batchsize, torch.float32)