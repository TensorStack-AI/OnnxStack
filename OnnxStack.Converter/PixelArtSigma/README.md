# OnnxStack.Converter

## Requirements
```bash
python -m pip install -r requirements.txt
```

## Usage
```bash
python convert.py --model_input "D:\Models\PixArt-Sigma-XL"
```

`--model_input`  - Safetensor model to convert

`--model_output`  - Output for converted ONNX model

`--clean`  - Clear convert/optimize model cache

`--tempDir`  - Directory for temp Olive files

`--only_unet`  - Only convert UNET model

