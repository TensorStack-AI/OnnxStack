# OnnxStack.Converter

## Requirements
```bash
python -m pip install -r requirements.txt
```

# ConvertDiffusersToOnnx
Convert Diffusers model to Onnx format

---

## Usage
```bash
python convertDiffusersToOnnx.py --input "D:\Models\FLUX_Diffusers"
```

## Options

- **`--input`**  
  Safetensor model to convert.

- **`--output`**  
  *(Optional)* Output for the converted ONNX model.  
  *Default:* `model_input\_onnx`

- **`--modules`**  
  *(Optional)* The modules to convert.  
  *Default:* `tokenizer,tokenizer_2,vae_encoder,vae_decoder,transformer,text_encoder,text_encoder_2`

- **`--conversion`**  
  *(Optional)* The conversion types to generate (optimized or unoptimized).  
  *Default:* `optimized`

- **`--clean`**  
  *(Optional)* Clear convert/optimize model cache.  
  *Default:* `false`

- **`--temp`**  
  *(Optional)* Directory for temp Olive files.  
  *Default:* `\temp`

-----------------------------------------------



# ConvertSafetensorToDiffusers
Convert Safetensor file to Diffusers format

---

## Usage
```bash
python convertSafetensorToDiffusers.py --input "D:\Models\flux1-schnell-fp8.safetensors" --output "D:\Models\FLUX_Diffusers"
```

## Options

- **`--input`**  
  Safetensor model to convert.

- **`--output`**  
  (Optional) Output for the converted ONNX model.  
  *Default:* `model_input\_onnx`

- **`--cache`**  
  (Optional) The path to you Diffusers cache if required.  
  *Default:* `None`

- **`--lora`**  
  (Optional) LoRA file and weight (file:weight)

- **`--modules`**  
  (Optional) The modules to convert.  
  *Default:* `tokenizer,tokenizer_2,vae_encoder,vae_decoder,transformer,text_encoder,text_encoder_2`