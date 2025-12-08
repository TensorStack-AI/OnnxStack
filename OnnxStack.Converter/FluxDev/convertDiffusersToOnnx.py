import os
import argparse
import json
import shutil
import warnings
from pathlib import Path
import onnx
import uuid
from onnx import helper
from onnx import TensorProto
from olive.workflows import run as olive_run
from olive.model import ONNXModelHandler


def optimize(script_dir: str, model_input: str, model_output: Path, submodel_names: list[str]):
    model_info = {}
    model_dir = model_input
   
    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}...")
        if submodel_name in ("tokenizer", "tokenizer_2", "tokenizer_3"):
            print(f"Optimizing {submodel_name} complete.")
            continue

        olive_config = None
        with (script_dir / f"config_{submodel_name}.json").open() as fin:
            olive_config = json.load(fin)

        olive_config["input_model"]["config"]["model_path"] = model_dir
        run_res = olive_run(olive_config)
        save_onnx_submodel(script_dir, submodel_name, model_info)
        print(f"Optimizing {submodel_name} complete.")

    save_onnx_models(model_dir, model_info, model_output, submodel_names)
    return model_info


def save_onnx_models(model_dir, model_info, model_output, submodel_names):
    model_dir = Path(model_dir)
    model_output.mkdir(parents=True, exist_ok=True)
    
    for submodel_name in submodel_names:
        print(f"Saving {submodel_name} model...")
        if submodel_name in ("tokenizer", "tokenizer_2", "tokenizer_3"):
            if os.path.exists(model_dir / submodel_name):
                shutil.copytree(model_dir / submodel_name, model_output / submodel_name, ignore=shutil.ignore_patterns("*tokenizer_config.json"))
            continue
        
        dst_dir = model_output / submodel_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        # model.onnx & model.onnx.data
        src_path = model_info[submodel_name]["path"]
        src_data_path = src_path.parent / "model.onnx.data"

        if submodel_name == "transformer":
            postProcess(src_path, src_data_path)

        shutil.copy(src_path, dst_dir)
        if os.path.exists(src_data_path):
            shutil.copy(src_data_path, dst_dir)

    print(f"Model Output: {model_output}")


def save_onnx_submodel(script_dir, submodel_name, model_info):
    footprints_file_path = (script_dir / ".olive-cache" / "models" / submodel_name / "footprints.json")
    with footprints_file_path.open("r") as footprint_file:
        footprints = json.load(footprint_file)

        optimizer_footprint = None
        for footprint in footprints.values():
            if footprint["from_pass"] == "OnnxFloatToFloat16":
                optimizer_footprint = footprint
            elif footprint["from_pass"] == "OnnxPeepholeOptimizer":
                optimizer_footprint = footprint
            elif footprint["from_pass"] == "OrtTransformersOptimization":
                optimizer_footprint = footprint
        assert optimizer_footprint

        optimized_olive_model = ONNXModelHandler(**optimizer_footprint["model_config"]["config"])
        model_info[submodel_name] = {
            "path": Path(optimized_olive_model.model_path)
        }


def postProcess(modelFile, modelDataFile):
    print("transformer post process...")
    model = onnx.load(modelFile)
    einsum_node_names = ["/pos_embed/Einsum","/pos_embed/Einsum_1","/pos_embed/Einsum_2"]
    for einsum_node_name in einsum_node_names:
        for node in model.graph.node:
            if node.op_type == "Einsum" and node.name == einsum_node_name:
                einsum_node = node
                #print(f"Found Einsum node: {node.name}")

        input_to_change = einsum_node.input[0]
        for input_info in model.graph.input:
            if input_info.name == input_to_change:
                input_info.type.tensor_type.elem_type = TensorProto.DOUBLE  # Change to FLOAT64 (DOUBLE)
                #print("input_info updated")
        
        # Create the Cast node
        cast_output_name = input_to_change + "_cast_to_float64"
        cast_node = helper.make_node(
            'Cast',
            inputs=[input_to_change],
            outputs=[cast_output_name],
            to=TensorProto.DOUBLE  # Cast to FLOAT64 (DOUBLE)
        )

        # Add the Cast node to the graph
        model.graph.node.append(cast_node)

        # Replace the original input to Einsum with the cast output
        einsum_node.input[0] = cast_output_name
        #print("Tensor updated")


    # Loop through the consumer nodes
    consumer_node_names = ["/pos_embed/Cos", "/pos_embed/Sin", "/pos_embed/Cos_1", "/pos_embed/Sin_1", "/pos_embed/Cos_2", "/pos_embed/Sin_2"] 
    for consumer_node_name in consumer_node_names:
        for node in model.graph.node:
            if node.name == consumer_node_name:
                consumer_node = node
                #print(f"Found consumer node: {node.name}")
                
                for i, input_name in enumerate(consumer_node.input):
                  
                    # Create the Cast node to convert the input from float64 to float16
                    cast_output_name = input_name + "_cast_to_float16_" + str(uuid.uuid4())[:8] #unique name
                    cast_node = helper.make_node(
                        'Cast',
                        inputs=[input_name],
                        outputs=[cast_output_name],
                        to=TensorProto.FLOAT16  # Cast to float16
                    )
                    
                    # Add the Cast node to the graph
                    model.graph.node.append(cast_node)
                    
                    # Update the consumer node's input to use the casted output
                    consumer_node.input[i] = cast_output_name
                    #print("Tensor updated")

    # Delete old, save new
    os.remove(modelFile)
    if os.path.exists(modelDataFile):
        os.remove(modelDataFile)
    onnx.save(model, modelFile, save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx.data")
    print("transformer post process complete.")


def clean(script_dir):
    shutil.rmtree(script_dir / ".olive-cache", ignore_errors=True)


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--modules", default="tokenizer,tokenizer_2,text_encoder,text_encoder_2,vae_encoder,vae_decoder,transformer", help="The modules to convert `tokenizer,tokenizer_2,text_encoder,text_encoder_2,vae_encoder,vae_decoder,transformer`")
    parser.add_argument("--clean", default=False, action="store_true", help="Deletes the Olive cache")
    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)
    model_input = common_args.input
    model_output = common_args.output
    submodel_names = common_args.modules.split(",")
    script_dir = Path(__file__).resolve().parent

    if model_output is None:
        model_output = Path(model_input) / "_onnx"
        shutil.rmtree(model_output, ignore_errors=True)

    if common_args.clean:
        clean(script_dir)

    print('Olive Conversion - Flux Schnell Model')
    print('--------------------------------------')
    print(f'Input: {model_input}')
    print(f'Output: {model_output}')
    print(f'Modules: {submodel_names}')
    print('--------------------------------------')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimize(script_dir, model_input, model_output, submodel_names)

    # clean(script_dir)
    print('Olive Flux Schnell Conversion Complete.')


if __name__ == "__main__":
    main()
