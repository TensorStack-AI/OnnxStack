# OnnxStack.Converter

## Requirements
```bash
python -m pip install -r requirements.txt
```

## Usage Safetensor
```bash
python convertSafetensorToOnnx.py --input "D:\Models\stable-diffusion-xl-base-1.0.safetensors"
```

## Usage Diffusers
```bash
python convertDiffusersToOnnx.py --input "D:\Models\stable-diffusion-xl-base-1.0"
```


`--input`  - Diffuser or Safetensors model to convert

`--output`  - (optional) Output for converted ONNX model

`--modules`  - (optional) The modules to convert (optional)

`--clean`  -  (optional) Clear convert/optimize model cache (optional)

`--vae_fp16_fix`  -  (optional) Enable the VAEncoder FP16 fix (https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)