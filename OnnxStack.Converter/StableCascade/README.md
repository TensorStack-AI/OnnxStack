# OnnxStack.Converter

Note: Prior model currently is still float32, attempts to convert to float16 have resulted in a black image result, all other models are float16


## Requirements
```bash
python -m pip install -r requirements.txt
```

## Usage Safetensor
```bash
python convertSafetensorToOnnx.py --input "D:\Models\stable-cascade.safetensors"
```

## Usage Diffusers
```bash
python convertDiffusersToOnnx.py --input "D:\Models\stable-cascade"
```


`--input`  - Diffuser or Safetensors model to convert

`--output`  - (optional) Output for converted ONNX model

`--modules`  - (optional) The modules to convert (optional)

`--clean`  -  (optional) Clear convert/optimize model cache (optional)