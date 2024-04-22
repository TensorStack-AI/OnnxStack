# OnnxStack.Converter

## Requirements
```bash
pip install onnxruntime-directml
pip install olive-ai[directml]
python -m pip install -r requirements.txt
```

## Usage
```bash
convert.py --optimize --model_input '..\stable-cascade' --image_encoder
```
`--optimize`  - Run the model optimization

`--model_input`  - Safetensor model to convert

`--model_output`  - Output for converted ONNX model (NOTE: This folder is deleted before each run)

`--image_encoder`  - Convert the optional image encoder
