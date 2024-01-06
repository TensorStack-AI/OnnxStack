# Example - Console App

## Getting Started

### 1.) Download the required `.onnx` models:


  * https://huggingface.co/rocca/swin-ir-onnx
    * This is the upscaler model.
  * https://huggingface.co/runwayml/stable-diffusion-v1-5
  * https://huggingface.co/softwareweaver/InstaFlow-0.9B-Olive-Onnx
  * https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
  * https://huggingface.co/softwareweaver/stable-diffusion-xl-base-1.0-Olive-Onnx

Note: Ensure you run ```git lfs install``` before cloning the repository to ensure the models are downloaded correctly.


### 2.) Update the paths in the `appsettings.json` within the `/OnnxStack.Console/` project:


* Update the paths in `appsettings.json` to point to the downloaded models
  * If the downloaded model repository does not contain a `Tokenizer` or `Tokenizer2` `.onnx` file, leave the path empty. 
    * Note: When the path is empty, OnnxStack will use it's own 'built-in' tokenizer called `cliptokenizer.onnx`
  * Example with empty path:
```json    
            {
            "Type": "Tokenizer",
            "OnnxModelPath": ""
            }
```


### FAQ

* **Q:** My `.GIF` is flashing, is it supposed to do this?
* **A:** The `.GIF` area has been deprecated in favor of the 'video' features that are being added.  Please ignore this example for now.


 