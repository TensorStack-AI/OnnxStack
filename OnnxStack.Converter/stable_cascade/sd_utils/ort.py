# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import onnxruntime as ort
from diffusers import OnnxRuntimeModel, StableCascadePriorPipeline
from onnxruntime import __version__ as OrtVersion
from packaging import version

from olive.model import ONNXModelHandler

# ruff: noqa: TID252, T201


def update_cuda_config(config: Dict):
    if version.parse(OrtVersion) < version.parse("1.17.0"):
        # disable skip_group_norm fusion since there is a shape inference bug which leads to invalid models
        config["passes"]["optimize_cuda"]["config"]["optimization_options"] = {"enable_skip_group_norm": False}
    config["pass_flows"] = [["convert", "optimize_cuda"]]
    config["systems"]["local_system"]["config"]["accelerators"][0]["execution_providers"] = ["CUDAExecutionProvider"]
    return config


def validate_args(args, provider):
    ort.set_default_logger_severity(4)
    if args.static_dims:
        print(
            "WARNING: the --static_dims option is deprecated, and static shape optimization is enabled by default. "
            "Use --dynamic_dims to disable static shape optimization."
        )

    validate_ort_version(provider)


def validate_ort_version(provider: str):
    if provider == "dml" and version.parse(OrtVersion) < version.parse("1.16.0"):
        print("This script requires onnxruntime-directml 1.16.0 or newer")
        sys.exit(1)
    elif provider == "cuda" and version.parse(OrtVersion) < version.parse("1.17.0"):
        if version.parse(OrtVersion) < version.parse("1.16.2"):
            print("This script requires onnxruntime-gpu 1.16.2 or newer")
            sys.exit(1)
        print(
            f"WARNING: onnxruntime {OrtVersion} has known issues with shape inference for SkipGroupNorm. Will disable"
            " skip_group_norm fusion. onnxruntime-gpu 1.17.0 or newer is strongly recommended!"
        )


def save_optimized_onnx_submodel(submodel_name, provider, model_info):
    footprints_file_path = (
        Path(__file__).resolve().parents[1] / "footprints" / f"{submodel_name}_gpu-{provider}_footprints.json"
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

        unoptimized_olive_model = ONNXModelHandler(**conversion_footprint["model_config"]["config"])
        optimized_olive_model = ONNXModelHandler(**optimizer_footprint["model_config"]["config"])

        model_info[submodel_name] = {
            "unoptimized": {
                "path": Path(unoptimized_olive_model.model_path),
                "data": Path(unoptimized_olive_model.model_path + ".data"),
            },
            "optimized": {
                "path": Path(optimized_olive_model.model_path),
                "data": Path(optimized_olive_model.model_path + ".data"),
            },
        }

        print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")
        print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")


def save_onnx_pipeline(
    model_info, model_output, pipeline, submodel_names
):
    # Save the unoptimized models in a directory structure that the diffusers library can load and run.
    # This is optional, and the optimized models can be used directly in a custom pipeline if desired.
    # print("\nCreating ONNX pipeline...")
   
    # TODO: Create OnnxStableCascadePipeline

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    print("Copying optimized models...")
    for passType in ["optimized", "unoptimized"]:
        model_dir = model_output / passType
        for submodel_name in submodel_names:
            src_path = model_info[submodel_name][passType]["path"] # model.onnx
            src_data_path = model_info[submodel_name][passType]["data"]# model.onnx.data

            dst_path = model_dir / submodel_name
            if not os.path.exists(dst_path):
                os.makedirs(dst_path, exist_ok=True)

            shutil.copyfile(src_path, dst_path / "model.onnx")
            if os.path.exists(src_data_path):
                shutil.copyfile(src_data_path, dst_path / "model.onnx.data")
        
    print(f"The converted model is located here: {model_output}")
