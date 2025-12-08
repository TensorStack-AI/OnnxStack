import torch
import torch.nn as nn
from typing import Union, Tuple
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


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
# CONTENT FILTER
# -----------------------------------------------------------------------------

class ContentFilterModel(StableDiffusionSafetyChecker):
    def forward(
        self,
        clip_input: torch.FloatTensor
    ) -> Union[StableDiffusionSafetyChecker, Tuple]:

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        special_scores = special_cos_dist - self.special_care_embeds_weights
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        return concept_scores
      

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


def content_filter_inputs(batchsize, torch_dtype):
    return { "clip_input": torch.rand((1, 3, 224, 224), dtype=torch_dtype) }


def content_filter_load(model_name):
    model = ContentFilterModel.from_pretrained(model_name, subfolder="safety_checker")
    return model


def content_filter_conversion_inputs(model=None):
    return tuple(content_filter_inputs(1, torch.float32).values())


def safety_checker_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(content_filter_inputs, batchsize, torch.float16)
