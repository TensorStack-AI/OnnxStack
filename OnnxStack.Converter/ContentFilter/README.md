# OnnxStack.Converter

## Requirements
```bash
pip install onnxruntime-directml
pip install olive-ai[directml]
python -m pip install -r requirements.txt
```

## Usage
```bash
python convert.py --model_input "D:\Models\stable-diffusion-v1-5" --controlnet
```

`--model_input`  - Safetensor model to convert

`--model_output`  - Output for converted ONNX model

`--clean`  - Clear convert/optimize model cache

`--tempDir`  - Directory for temp Olive files

`--only_unet`  - Only convert UNET model

`--controlnet`  - Create a ControlNet enabled Unet model