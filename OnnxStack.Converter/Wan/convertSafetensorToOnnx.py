import subprocess
import os
import sys
import argparse
import shutil
from pathlib import Path
from diffusers import StableDiffusion3Pipeline


def save_diffusers(safetensorFile: str, output_dir: str):
    pipeline = StableDiffusion3Pipeline.from_single_file(safetensorFile, config="stabilityai/stable-diffusion-3-medium-diffusers")
    pipeline.save_pretrained(output_dir)


def serialize_args(common_args: object, diffusers_output: str):
   
    if common_args.output is None:  
        filename = os.path.splitext(os.path.basename(common_args.input))[0]
        common_args.output = Path(common_args.input).parent / filename
        shutil.rmtree(common_args.output, ignore_errors=True)

    common_args.input = diffusers_output

    arg_list = []
    for key, value in vars(common_args).items():
        if isinstance(value, bool):  # Handle flags
            if value:
                arg_list.append(f"--{key}")
        else:
            arg_list.extend([f"--{key}", str(value)])
    return arg_list


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")
    parser.add_argument("--input", default="stable-diffusion-xl", type=str)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--modules", default="tokenizer,text_encoder,vae_decoder,transformer", help="The modules to convert `tokenizer,text_encoder,vae_decoder,transformer`")
    parser.add_argument("--conversion", default="optimized", help="Type of conversion: optimized or unoptimized")
    parser.add_argument("--clean", default=False, action="store_true", help="Deletes the Olive cache")
    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)
    script_dir = Path(__file__).resolve().parent

    print('Diffusers Conversion - SD3 Model')
    print('--------------------------------------')
    print(f'Input: {common_args.input}')
    print('--------------------------------------')
    diffusers_output = (script_dir / ".olive-cache" / "diffusers")
    save_diffusers(common_args.input, diffusers_output)
    print('Diffusers Conversion Compete.\n')

    # convertDiffusersToOnnx
    subprocess.run([sys.executable, "convertDiffusersToOnnx.py"] + serialize_args(common_args, diffusers_output))


if __name__ == "__main__":
    main()
