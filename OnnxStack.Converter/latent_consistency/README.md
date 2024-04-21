# OnnxStack.Converter

## Requirements
```bash
pip install onnxruntime-directml
pip install olive-ai[directml]
python -m pip install -r requirements.txt
```

## Usage
```bash
convert.py --model_input "D:\Models\LCM_Dreamshaper_v7" --controlnet
```

`--model_input`  - Safetensor model to convert

`--model_output`  - Output for converted ONNX model

`--controlnet`  - Create a ControlNet enabled Unet model

`--clean`  - Clear convert/optimize model cache

`--tempDir`  - Directory for temp Olive files