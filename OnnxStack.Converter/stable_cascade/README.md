# OnnxStack.Converter

## Requirements
```bash
pip install onnxruntime-directml
pip install olive-ai[directml]
python -m pip install -r requirements.txt
```

## Usage
```bash
python convert.py --model_input "D:\Models\stable-cascade"  --image_encoder
```

`--model_input`  - Safetensor model to convert

`--model_output`  - Output for converted ONNX model

`--clean`  - Clear convert/optimize model cache

`--tempDir`  - Directory for temp Olive files

`--only_unet`  - Only convert UNET model

`--image_encoder`  - Convert the optional image encoder
