## Troubleshooting

 - I'm running on linux but it's not working citing:`The ONNX Runtime extensions library was not found`?
   - It's having a problem loading `libortextensions.so`
   - From the project root run `find -name "libortextensions.so"` to locate that file
   - Then run `ldd libortextensions.so` against it to see what dependencies it needs versus what your system has.
   - It has a dependency on SSL 1.1 which was removed from Ubuntu based OSes and causes this error.
   - It can be remedied by manually installing the dependency. 
   - See: https://stackoverflow.com/questions/72133316/libssl-so-1-1-cannot-open-shared-object-file-no-such-file-or-directory
 - I've installed `Microsoft.ML.OnnxRuntime` and `Microsoft.ML.OnnxRuntime.Gpu` into my project and set the execution provider to `Cuda`, but it's complaining it can't find an entry point for CUDA?
   - `System.EntryPointNotFoundException : Unable to find an entry point named 'OrtSessionOptionsAppendExecutionProvider_CUDA' in shared library 'onnxruntime'`
   - Adding both `Microsoft.ML.OnnxRuntime` AND `Microsoft.ML.OnnxRuntime.Gpu` at the same time causes this.
   - Remove `Microsoft.ML.OnnxRuntime` and try again.
 - I'm trying to run via CUDA execution provider but it's complaining about missing `libcublaslt11`, `libcublas11`, or `libcudnn8`?
   - Aside from just the NVIDIA Drivers you also need to install CUDA, and cuDNN.
