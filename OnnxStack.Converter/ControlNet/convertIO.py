import os
import onnx
import argparse
from pathlib import Path
from onnx import helper, TensorProto

def convert(model_input: Path, model_output: Path, external_data: bool):
    model = onnx.load(model_input)

    castCount = 0
    graph = model.graph

    # Inputs
    input_cast_nodes = []
    for input_tensor in graph.input:
        original_dtype = input_tensor.type.tensor_type.elem_type 
        
        if original_dtype == TensorProto.FLOAT16: 
            
            input_tensor.type.tensor_type.elem_type = TensorProto.FLOAT  
            cast_output_name = f"{input_tensor.name}_iocast_{castCount}"  
            cast_node = helper.make_node(
                "Cast",
                name=cast_output_name,
                inputs=[input_tensor.name],
                outputs=[cast_output_name],
                to=original_dtype
            )

            input_cast_nodes.append(cast_node)
            for node in graph.node:
                for i, input_name in enumerate(node.input):
                    if input_name == input_tensor.name:
                        node.input[i] = cast_node.output[0]

            castCount += 1
            print(f"Input: {cast_output_name}")

    graph.node.extend(input_cast_nodes)
    
    # Outputs
    for output_tensor in graph.output:
        original_dtype = output_tensor.type.tensor_type.elem_type

        if original_dtype == TensorProto.FLOAT16:
            output_tensor.type.tensor_type.elem_type = TensorProto.FLOAT
            producing_node = None
            for node in graph.node:
                if output_tensor.name in node.output:
                    producing_node = node
                    break

            if producing_node:
                cast_output_name = f"{output_tensor.name}_iocast_{castCount}"  
                cast_node = helper.make_node(
                    "Cast",
                    name=cast_output_name,
                    inputs=[cast_output_name], 
                    outputs=[producing_node.output[0]], 
                    to=TensorProto.FLOAT 
                )

                graph.node.append(cast_node)
                producing_node.output[0] = cast_output_name
                castCount += 1
                print(f"Output: {cast_output_name}")

    print("Saving Onnx Model...")
    onnx.save(model, model_output,save_as_external_data=external_data,all_tensors_to_one_file=external_data, location=F"{model_output.name}.data")



def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--external_data", action="store_true")
    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)
    model_input = common_args.input
    model_output = common_args.output
    external_data= common_args.external_data
    script_dir = Path(__file__).resolve().parent

    if model_output is None:
        model_output = Path(model_input).parent / "converted.onnx"
        model_output_data = f"{model_output}.data"
        if os.path.exists(model_output):
            os.remove(model_output)
        if os.path.exists(model_output_data):
            os.remove(model_output_data)

    print('IO/16 to IO/32 Conversion')
    print('--------------------------------------')
    print(f'Input: {model_input}')
    print(f'Output: {model_output}')
    print(f'External Data: {external_data}')
    print('--------------------------------------')

    convert(model_input, model_output, external_data)

    print('IO/16 to IO/32 Conversion Complete.')
        

if __name__ == "__main__":
    main()
