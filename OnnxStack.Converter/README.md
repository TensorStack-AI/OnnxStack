# OnnxStack.Converter

## Requirements
```bash
pip install onnxruntime-directml
pip install olive-ai[directml]
python -m pip install -r requirements.txt
```

## Usage
```bash
convert.py --optimize --model_input '..\stable-diffusion-v1-5' --model_output '..\converted' --controlnet
```
`--optimize`  - Run the model optimization

`--model_input`  - Safetensor model to convert

`--model_output`  - Output for converted ONNX model (NOTE: This folder is deleted before each run)

`--controlnet`  - Create a ControlNet enabled Unet model