# Example - Console App

## Getting Started

### 1.) Download the required `.onnx` models:

**Method 1**

  Download the required models from the following repositories:

  * https://huggingface.co/rocca/swin-ir-onnx
    * This is the upscaler model.
  * https://huggingface.co/runwayml/stable-diffusion-v1-5
  * https://huggingface.co/softwareweaver/InstaFlow-0.9B-Olive-Onnx
  * https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
  * https://huggingface.co/softwareweaver/stable-diffusion-xl-base-1.0-Olive-Onnx


 Note: Ensure you run ```git lfs install``` before cloning the repository to ensure the models are downloaded correctly.



 

 

**Method 2**

 Download Files

Use the included 'download-models.bat' file to download the required models.
* If your on mac or linux, you can open the `download-models.bat` file and run each command manually if you are unable to run the `.bat` file.
  * The `download-models.bat` will create a `onnx-models/` folder in current directory. Then it will download the required models into the `onnx-models/` folder.
  * If you are unable to run the `.bat` file or have issues, you can open the `download-models.bat` file and run each command manually.
  * The `.bat` file will not attempt to 're-download' or continue if the file already exists. If you need to re-download the models, delete the `onnx-models/` folder and run the `.bat` file again.


### 2.) If necessary, switch to the correct Stable Diffusion 1.5 branch

**Stable Diffusion 1.5 - Get the .onnx files after download.**

* After you download the `Stable Diffusion 1.5` model at: https://huggingface.co/runwayml/stable-diffusion-v1-5, check to see if any of the folders contain `.onnx` files.
  * If none contain `.onnx` files, you will need to switch to the `.onnx` branch.
    * If you have already checked out the `onnx` branch and have the `.onnx` files from the `Stable Diffusion 1.5` repository, you can skip this step.


We need to switch to the `onnx` branch to get the `.onnx` files.  
You can do this anyway you would like, or just run the following commands:
* Open a new command prompt in `onnx-models/stable-diffusion-v1-5/`
```bash
git fetch origin
git checkout origin/onnx
```
_It might take some time to switch branches and download all of the `.onnx` files pending your internet speed._
 


### 3.) Update the paths in the `appsettings.json` within the `/OnnxStack.Console/` project:


* Update the paths in `appsettings.json` to point to the downloaded models.  Update all the paths to point to the correct location of the downloaded models.
  * If the downloaded model repository does not contain a `Tokenizer` or `Tokenizer2` `.onnx` file, leave the path empty. 
    * Note: When the path is empty, OnnxStack will use it's own 'built-in' _default_ tokenizer called `cliptokenizer.onnx`
  * Example with empty path:
```json    
  {
  "Type": "Tokenizer",
  "OnnxModelPath": ""
  },
  {
  "Type": "Tokenizer2",
  "OnnxModelPath": ""
  }
```


### FAQ

* **Q:** My `.GIF` is flashing, is it supposed to do this?
* **A:** The `.GIF` area has been deprecated in favor of the 'video' features that are being added.  Please ignore this example for now.


 