# OnnxStack.Converter

## Requirements
```bash
python -m pip install -r requirements.txt
```

## Usage Safetensor
```bash
python convertSafetensorToOnnx.py --input "D:\Models\sd3_medium.safetensors"
```

## Usage Diffusers
```bash
python convertDiffusersToOnnx.py --input "D:\Models\stable-diffusion-3-medium"
```


`--input`  - Diffuser or Safetensors model to convert

`--output`  - (optional) Output for converted ONNX model

`--modules`  - (optional) The modules to convert (optional)

`--clean`  -  (optional) Clear convert/optimize model cache (optional)