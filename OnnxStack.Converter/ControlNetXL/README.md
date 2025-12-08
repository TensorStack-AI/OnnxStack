# OnnxStack.Converter

## Requirements
```bash
python -m pip install -r requirements.txt
```

## Usage
```bash
python convertToOnnx.py --input "D:\Models\ControlNet"
```

`--input`  - Diffuser or Safetensors model to convert

`--output`  - (optional) Output for converted ONNX model

`--modules`  - (optional) The modules to convert (optional)

`--clean`  -  (optional) Clear convert/optimize model cache (optional)