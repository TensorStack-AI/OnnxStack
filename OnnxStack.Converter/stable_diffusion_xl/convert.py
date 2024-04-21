# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import config
import torch
from diffusers import DiffusionPipeline
from packaging import version
import onnx
from onnx import helper, onnx_pb as onnx_proto
from onnxruntime_extensions import make_onnx_model
from olive.common.utils import set_tempdir
from olive.workflows import run as olive_run
from olive.model import ONNXModelHandler


def optimize(
    script_dir: str,
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

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)

    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print("Download stable diffusion PyTorch pipeline...")
    pipeline = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float32, **{"local_files_only": True})
    config.vae_sample_size = pipeline.vae.config.sample_size
    config.cross_attention_dim = pipeline.unet.config.cross_attention_dim
    config.unet_sample_size = pipeline.unet.config.sample_size

    model_info = {}
    submodel_names = ["tokenizer", "tokenizer_2", "vae_encoder", "vae_decoder", "unet", "text_encoder", "text_encoder_2"]

    if controlnet:
        submodel_names.append("controlnet")

    for submodel_name in submodel_names:
        if submodel_name == "tokenizer" or submodel_name == "tokenizer_2":
            save_onnx_tokenizer_model(script_dir, model_input, submodel_name, model_info)
            continue

        print(f"\nOptimizing {submodel_name}")
        olive_config = None
        with (script_dir / f"config_{submodel_name}.json").open() as fin:
            olive_config = json.load(fin)

        olive_config["input_model"]["config"]["model_path"] = model_dir
        run_res = olive_run(olive_config)
        save_onnx_submodel(script_dir, submodel_name, model_info, provider)

    save_onnx_Models(model_dir, model_info, model_output, submodel_names)
    return model_info


def save_onnx_Models(model_dir, model_info, model_output, submodel_names):
    model_dir = Path(model_dir)
    for conversion_type in ["optimized", "unoptimized"]:

        conversion_dir = model_output / conversion_type
        conversion_dir.mkdir(parents=True, exist_ok=True)

        # Copy the config and other files required by some applications
        model_index_path = model_dir / "model_index.json"
        if os.path.exists(model_index_path):
            shutil.copy(model_index_path, conversion_dir)
        if os.path.exists(model_dir / "tokenizer"):
            shutil.copytree(model_dir / "tokenizer", conversion_dir / "tokenizer")
        if os.path.exists(model_dir / "tokenizer_2"):
            shutil.copytree(model_dir / "tokenizer_2", conversion_dir / "tokenizer_2")
        if os.path.exists(model_dir / "scheduler"):
            shutil.copytree(model_dir / "scheduler", conversion_dir / "scheduler")

        # Save models files
        for submodel_name in submodel_names:
            print(f"Saving {conversion_type} {submodel_name} model...")
            dst_dir = conversion_dir / submodel_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Copy the model.onnx
            # model.onnx
            src_path = model_info[submodel_name][conversion_type]["path"]
            shutil.copy(src_path, dst_dir)

            # Copy the model.onnx.data if it exists
            src_data_path = src_path.parent / "model.onnx.data"
            if os.path.exists(src_data_path):
                shutil.copy(src_data_path, dst_dir)

            # Copy the model config.json if it exists
            src_dir = model_dir / submodel_name
            if submodel_name == "controlnet":
                src_dir = model_dir / "unet"
            if submodel_name == "vae_encoder" or submodel_name == "vae_decoder":
                src_dir = model_dir / "vae"
            config_path = src_dir / "config.json"
            if os.path.exists(config_path):
                shutil.copy(config_path, dst_dir)

    print(f"The optimized models located here: {model_output}\\optimized")
    print(f"The unoptimized models located here: {model_output}\\unoptimized")


def save_onnx_submodel(script_dir, submodel_name, model_info, provider):
    footprints_file_path = (
        script_dir / "footprints" /
        f"{submodel_name}_gpu-{provider}_footprints.json"
    )
    with footprints_file_path.open("r") as footprint_file:
        footprints = json.load(footprint_file)

        conversion_footprint = None
        optimizer_footprint = None
        for footprint in footprints.values():
            if footprint["from_pass"] == "OnnxConversion":
                conversion_footprint = footprint
            elif footprint["from_pass"] == "OrtTransformersOptimization":
                optimizer_footprint = footprint

        assert conversion_footprint
        assert optimizer_footprint

        unoptimized_olive_model = ONNXModelHandler(
            **conversion_footprint["model_config"]["config"])
        optimized_olive_model = ONNXModelHandler(
            **optimizer_footprint["model_config"]["config"])

        model_info[submodel_name] = {
            "unoptimized": {"path": Path(unoptimized_olive_model.model_path)},
            "optimized": {"path": Path(optimized_olive_model.model_path)}
        }

        print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")
        print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")


def save_onnx_tokenizer_model(script_dir, model_dir, submodel_name, model_info, max_length=-1, attention_mask=True, offset_map=False):
    model_dir = Path(model_dir)
    vocab_file = model_dir / submodel_name / "vocab.json"
    merges_file = model_dir / submodel_name / "merges.txt"

    input1 = helper.make_tensor_value_info('string_input', onnx_proto.TensorProto.STRING, [None])
    output1 = helper.make_tensor_value_info('input_ids', onnx_proto.TensorProto.INT64, ["batch_size", "num_input_ids"])
    output2 = helper.make_tensor_value_info('attention_mask', onnx_proto.TensorProto.INT64, ["batch_size", "num_attention_masks"])
    output3 = helper.make_tensor_value_info('offset_mapping', onnx_proto.TensorProto.INT64, ["batch_size", "num_offsets", 2])

    inputs = [input1]
    outputs = [output1]
    output_names = ["input_ids"]
    if attention_mask:
        if offset_map:
            outputs.append([output2, output3])
            output_names.append(['attention_mask', 'offset_mapping'])
        else:
            outputs.append(output2)
            output_names.append('attention_mask')

    node = [helper.make_node
            (
                'CLIPTokenizer',
                ['string_input'],
                output_names,
                vocab=get_file_content(vocab_file),
                merges=get_file_content(merges_file),
                name='bpetok',
                padding_length=max_length,
                domain='ai.onnx.contrib'
            )]
    graph = helper.make_graph(node, 'main_graph', inputs, outputs)
    model = make_onnx_model(graph)

    output_dir = script_dir / "cache" / submodel_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'model.onnx'
    onnx.save(model, output_path)
    model_info[submodel_name] = {
        "optimized": {"path": output_path},
        "unoptimized": {"path": output_path}
    }


def get_file_content(path):
    with open(path, "rb") as file:
        return file.read()


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")
    parser.add_argument("--model_input", default="stable-diffusion-v1-5", type=str)
    parser.add_argument("--model_output", default=None, type=Path)
    parser.add_argument("--controlnet", action="store_true", help="Create ControlNet Unet Model")
    parser.add_argument("--clean", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--tempdir", default=None, type=str, help="Root directory for tempfile directories and files")
    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)

    provider = "dml"
    model_input = common_args.model_input
    model_output = common_args.model_output
    script_dir = Path(__file__).resolve().parent

    if model_output is None:
        model_output = Path(model_input) / "_onnx"
        shutil.rmtree(model_output, ignore_errors=True)

    if common_args.tempdir is None:
        common_args.tempdir = script_dir / "temp"

    if common_args.clean:
        shutil.rmtree(script_dir / "temp", ignore_errors=True)
        shutil.rmtree(script_dir / "cache", ignore_errors=True)
        shutil.rmtree(script_dir / "footprints", ignore_errors=True)
  
    set_tempdir(common_args.tempdir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimize(script_dir, common_args.model_input,
                 model_output, provider, common_args.controlnet)


if __name__ == "__main__":
    main()
