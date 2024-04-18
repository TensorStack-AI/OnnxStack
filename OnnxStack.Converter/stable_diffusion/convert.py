# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict

import config
import torch
from diffusers import DiffusionPipeline
from packaging import version

from olive.common.utils import set_tempdir
from olive.workflows import run as olive_run


# pylint: disable=redefined-outer-name
# ruff: noqa: TID252, T201


def save_image(result, batch_size, provider, num_images, images_saved, image_callback=None):
    passed_safety_checker = 0
    for image_index in range(batch_size):
        if result.nsfw_content_detected is None or not result.nsfw_content_detected[image_index]:
            passed_safety_checker += 1
            if images_saved < num_images:
                output_path = f"result_{images_saved}.png"
                result.images[image_index].save(output_path)
                if image_callback:
                    image_callback(images_saved, output_path)
                images_saved += 1
                print(f"Generated {output_path}")
    print(f"Inference Batch End ({passed_safety_checker}/{batch_size} images).")
    print("Images passed the safety checker.")
    return images_saved


def run_inference_loop(
    pipeline,
    prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
    guidance_scale,
    strength: float,
    provider: str,
    image_callback=None,
    step_callback=None,
):
    images_saved = 0

    def update_steps(step, timestep, latents):
        if step_callback:
            step_callback((images_saved // batch_size) * num_inference_steps + step)

    while images_saved < num_images:
        print(f"\nInference Batch Start (batch size = {batch_size}).")

        kwargs = {}

        result = pipeline(
            [prompt] * batch_size,
            num_inference_steps=num_inference_steps,
            callback=update_steps if step_callback else None,
            height=image_size,
            width=image_size,
            guidance_scale=guidance_scale,
            **kwargs,
        )

        images_saved = save_image(result, batch_size, provider, num_images, images_saved, image_callback)


def update_config_with_provider(config: Dict, provider: str):
    if provider == "dml":
        # DirectML EP is the default, so no need to update config.
        return config
    elif provider == "cuda":
        from sd_utils.ort import update_cuda_config

        return update_cuda_config(config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def optimize(
    model_input: str,
    model_output: Path,
    provider: str,
    controlnet: bool
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        sys.exit(1)

    model_dir = model_input
    script_dir = Path(__file__).resolve().parent

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)
    shutil.rmtree(model_output, ignore_errors=True)


    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print("Download stable diffusion PyTorch pipeline...")
    pipeline = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float32, **{"local_files_only": True})
    config.vae_sample_size = pipeline.vae.config.sample_size
    config.cross_attention_dim = pipeline.unet.config.cross_attention_dim
    config.unet_sample_size = pipeline.unet.config.sample_size

    model_info = {}

    submodel_names = ["vae_encoder", "vae_decoder", "unet" , "text_encoder"]

    has_safety_checker = getattr(pipeline, "safety_checker", None) is not None

    if has_safety_checker:
        submodel_names.append("safety_checker")

    if controlnet:
        submodel_names.append("controlnet")

    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        with (script_dir / f"config_{submodel_name}.json").open() as fin:
            olive_config = json.load(fin)
        olive_config = update_config_with_provider(olive_config, provider)

        if submodel_name in ("unet", "controlnet", "text_encoder"):
            olive_config["input_model"]["config"]["model_path"] = model_dir
        else:
            # Only the unet & text encoder are affected by LoRA, so it's better to use the base model ID for
            # other models: the Olive cache is based on the JSON config, and two LoRA variants with the same
            # base model ID should be able to reuse previously optimized copies.
            olive_config["input_model"]["config"]["model_path"] = model_dir

        run_res = olive_run(olive_config)

        from sd_utils.ort import save_optimized_onnx_submodel

        save_optimized_onnx_submodel(submodel_name, provider, model_info)

    from sd_utils.ort import save_onnx_pipeline

    save_onnx_pipeline(
        has_safety_checker, model_info, model_output, pipeline, submodel_names
    )

    return model_info


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")

    parser.add_argument("--model_input", default="stable-diffusion-v1-5", type=str)
    parser.add_argument("--model_output", default="stable-diffusion-v1-5", type=Path)
    parser.add_argument("--controlnet",action="store_true", help="Create ControlNet Unet Model")
    parser.add_argument(
        "--provider", default="dml", type=str, choices=["dml", "cuda"], help="Execution provider to use"
    )
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    parser.add_argument(
        "--prompt",
        default=(
            "castle surrounded by water and nature, village, volumetric lighting, photorealistic, "
            "detailed and intricate, fantasy, epic cinematic shot, mountains, 8k ultra hd"
        ),
        type=str,
    )
    parser.add_argument(
        "--guidance_scale",
        default=7.5,
        type=float,
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance",
    )
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of steps in diffusion process")
    parser.add_argument("--tempdir", default=None, type=str, help="Root directory for tempfile directories and files")
    parser.add_argument(
        "--strength",
        default=1.0,
        type=float,
        help="Value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. "
        "Values that approach 1.0 enable lots of variations but will also produce images "
        "that are not semantically consistent with the input.",
    )
    parser.add_argument("--image_size", default=512, type=int, help="Width and height of the images to generate")

    return parser.parse_known_args(raw_args)


def parse_ort_args(raw_args):
    parser = argparse.ArgumentParser("ONNX Runtime arguments")

    parser.add_argument(
        "--static_dims",
        action="store_true",
        help="DEPRECATED (now enabled by default). Use --dynamic_dims to disable static_dims.",
    )
    parser.add_argument("--dynamic_dims", action="store_true", help="Disable static shape optimization")

    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)

    provider = common_args.provider
    model_input = common_args.model_input
    model_output = common_args.model_output

    script_dir = Path(__file__).resolve().parent


    if common_args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    guidance_scale = common_args.guidance_scale

    ort_args = None, None
    ort_args, extra_args = parse_ort_args(extra_args)

    if common_args.optimize or not model_output.exists():
        set_tempdir(common_args.tempdir)

        # TODO(jstoecker): clean up warning filter (mostly during conversion from torch to ONNX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            from sd_utils.ort import validate_args

            validate_args(ort_args, common_args.provider)
            optimize(common_args.model_input, common_args.model_output, common_args.provider, common_args.controlnet)

    if not common_args.optimize:
        model_dir = model_output / "F32" if common_args.test_unoptimized else model_output / "F16"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
      
            from sd_utils.ort import get_ort_pipeline

            pipeline = get_ort_pipeline(model_dir, common_args, ort_args, guidance_scale)
            run_inference_loop(
                pipeline,
                common_args.prompt,
                common_args.num_images,
                common_args.batch_size,
                common_args.image_size,
                common_args.num_inference_steps,
                guidance_scale,
                common_args.strength,
                provider=provider,
            )


if __name__ == "__main__":
    main()
