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

from packaging import version
from olive.common.utils import set_tempdir
from olive.workflows import run as olive_run
from olive.model import ONNXModelHandler


def optimize(
    script_dir: str,
    model_input: str,
    model_output: Path,
    provider: str
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        sys.exit(1)

    model_dir = model_input

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)

    model_info = {}
    submodel_names = ["sd", "sdxl", "sd3", "flux"]

    for submodel_name in submodel_names:
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

        unoptimized_olive_model = ONNXModelHandler(**conversion_footprint["model_config"]["config"])
        optimized_olive_model = ONNXModelHandler(**optimizer_footprint["model_config"]["config"])

        model_info[submodel_name] = {
            "unoptimized": {"path": Path(unoptimized_olive_model.model_path)},
            "optimized": {"path": Path(optimized_olive_model.model_path)}
        }

        print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")
        print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")
    parser.add_argument("--model_output", default="", type=str)
    parser.add_argument("--clean", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--tempdir", default=None, type=str, help="Root directory for tempfile directories and files")
    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)

    provider = "dml"
    model_output = common_args.model_output
    script_dir = Path(__file__).resolve().parent

    if common_args.tempdir is None:
        common_args.tempdir = script_dir / "temp"

    if common_args.clean:
        shutil.rmtree(script_dir / "temp", ignore_errors=True)
        shutil.rmtree(script_dir / "cache", ignore_errors=True)
        shutil.rmtree(script_dir / "footprints", ignore_errors=True)
  
    set_tempdir(common_args.tempdir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimize(script_dir, model_output, Path(model_output), provider)


if __name__ == "__main__":
    main()
