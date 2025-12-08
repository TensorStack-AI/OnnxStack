import torch
from diffusers import AutoencoderTiny


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
# SD
# -----------------------------------------------------------------------------

def sd_inputs(batchsize, torch_dtype):
    return {"latent_sample": torch.rand((1, 4, 64, 64), dtype=torch_dtype)}


def sd_load(model_name):
    model = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def sd_conversion_inputs(model=None):
    return tuple(sd_inputs(1, torch.float32).values())


def sd_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(sd_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# SDXL
# -----------------------------------------------------------------------------

def sdxl_inputs(batchsize, torch_dtype):
    return {"latent_sample": torch.rand((1, 4, 128, 128), dtype=torch_dtype)}


def sdxl_load(model_name):
    model = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def sdxl_conversion_inputs(model=None):
    return tuple(sdxl_inputs(1, torch.float32).values())


def sdxl_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(sdxl_inputs, batchsize, torch.float16)



# -----------------------------------------------------------------------------
# SD3
# -----------------------------------------------------------------------------

def sd3_inputs(batchsize, torch_dtype):
    return {"latent_sample": torch.rand((1, 16, 128, 128), dtype=torch_dtype)}


def sd3_load(model_name):
    model = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def sd3_conversion_inputs(model=None):
    return tuple(sd3_inputs(1, torch.float32).values())


def sd3_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(sd3_inputs, batchsize, torch.float16)



# -----------------------------------------------------------------------------
# FLUX
# -----------------------------------------------------------------------------

def flux_inputs(batchsize, torch_dtype):
    return {"latent_sample": torch.rand((1, 16, 128, 128), dtype=torch_dtype)}


def flux_load(model_name):
    model = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.float32)
    model.forward = model.decode
    return model


def flux_conversion_inputs(model=None):
    return tuple(flux_inputs(1, torch.float32).values())


def flux_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(flux_inputs, batchsize, torch.float16)