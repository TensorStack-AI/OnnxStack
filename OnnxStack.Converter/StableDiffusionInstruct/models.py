import config
import torch
from typing import Union, Tuple
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers.models.clip.modeling_clip import CLIPTextModel

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
# UNET
# -----------------------------------------------------------------------------
def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "sample": torch.rand((batchsize, 8, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, config.cross_attention_dim), dtype=torch_dtype),
    }
    return inputs


def unet_load(model_name):
    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    return model


def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)



# -----------------------------------------------------------------------------
# CONTROLNET - UNET
# -----------------------------------------------------------------------------
class ControlNetUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        down_block_0_additional_residual: torch.Tensor,
        down_block_1_additional_residual: torch.Tensor,
        down_block_2_additional_residual: torch.Tensor,
        down_block_3_additional_residual: torch.Tensor,
        down_block_4_additional_residual: torch.Tensor,
        down_block_5_additional_residual: torch.Tensor,
        down_block_6_additional_residual: torch.Tensor,
        down_block_7_additional_residual: torch.Tensor,
        down_block_8_additional_residual: torch.Tensor,
        down_block_9_additional_residual: torch.Tensor,
        down_block_10_additional_residual: torch.Tensor,
        down_block_11_additional_residual: torch.Tensor,
        mid_block_additional_residual: torch.Tensor,
    ) -> Union[UNet2DConditionModel, Tuple]:
        down_block_add_res = (
            down_block_0_additional_residual, down_block_1_additional_residual, down_block_2_additional_residual,
            down_block_3_additional_residual, down_block_4_additional_residual, down_block_5_additional_residual,
            down_block_6_additional_residual, down_block_7_additional_residual, down_block_8_additional_residual,
            down_block_9_additional_residual, down_block_10_additional_residual, down_block_11_additional_residual)
        return super().forward(
            sample = sample,
            timestep = timestep,
            encoder_hidden_states = encoder_hidden_states,
            down_block_additional_residuals = down_block_add_res,
            mid_block_additional_residual = mid_block_additional_residual,
            return_dict = False
        )

def controlnet_unet_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 8, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, config.cross_attention_dim), dtype=torch_dtype),
        "down_block_0_additional_residual": torch.rand((batchsize, 320, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "down_block_1_additional_residual": torch.rand((batchsize, 320, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "down_block_2_additional_residual": torch.rand((batchsize, 320, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "down_block_3_additional_residual": torch.rand((batchsize, 320, config.unet_sample_size // 2, config.unet_sample_size // 2), dtype=torch_dtype),
        "down_block_4_additional_residual": torch.rand((batchsize, 640, config.unet_sample_size // 2, config.unet_sample_size // 2), dtype=torch_dtype),
        "down_block_5_additional_residual": torch.rand((batchsize, 640, config.unet_sample_size // 2, config.unet_sample_size // 2), dtype=torch_dtype),
        "down_block_6_additional_residual": torch.rand((batchsize, 640, config.unet_sample_size // 4, config.unet_sample_size // 4), dtype=torch_dtype),
        "down_block_7_additional_residual": torch.rand((batchsize, 1280, config.unet_sample_size // 4, config.unet_sample_size // 4), dtype=torch_dtype),
        "down_block_8_additional_residual": torch.rand((batchsize, 1280, config.unet_sample_size // 4, config.unet_sample_size // 4), dtype=torch_dtype),
        "down_block_9_additional_residual": torch.rand((batchsize, 1280, config.unet_sample_size // 8, config.unet_sample_size // 8), dtype=torch_dtype),
        "down_block_10_additional_residual": torch.rand((batchsize, 1280, config.unet_sample_size // 8, config.unet_sample_size // 8), dtype=torch_dtype),
        "down_block_11_additional_residual": torch.rand((batchsize, 1280, config.unet_sample_size // 8, config.unet_sample_size // 8), dtype=torch_dtype),
        "mid_block_additional_residual": torch.rand((batchsize, 1280, config.unet_sample_size // 8, config.unet_sample_size // 8), dtype=torch_dtype)
    }


def controlnet_unet_load(model_name):
    model = ControlNetUNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    return model


def controlnet_unet_conversion_inputs(model):
    return tuple(controlnet_unet_inputs(1, torch.float32).values())


def controlnet_unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(controlnet_unet_inputs, batchsize, torch.float16)