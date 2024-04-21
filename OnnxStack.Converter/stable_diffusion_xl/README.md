# OnnxStack.Converter

## Requirements
```bash
pip install onnxruntime-directml
pip install olive-ai[directml]
python -m pip install -r requirements.txt
```

## Usage
```bash
python convert.py --model_input "D:\Models\stable-diffusion-xl-base-1.0" --controlnet
```

`--model_input`  - Safetensor model to convert

`--model_output`  - Output for converted ONNX model

`--controlnet`  - Create a ControlNet enabled Unet model

`--clean`  - Clear convert/optimize model cache

`--tempDir`  - Directory for temp Olive files


## Extra Requirements
To successfully optimize SDXL models you will need the patched `vae` from repository below otherwise you may get black image results

https://huggingface.co/madebyollin/sdxl-vae-fp16-fix

Replace `diffusion_pytorch_model.safetensors` in the SDXL `vae` folder with the one in the `sdxl-vae-fp16-fix` repo
