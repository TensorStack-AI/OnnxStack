using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.Console
{
    public static class Providers
    {
        public static OnnxExecutionProvider CPU()
        {
            return new OnnxExecutionProvider("CPU", configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL
                };

                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }

        public static OnnxExecutionProvider DirectML(int deviceId)
        {
            return new OnnxExecutionProvider("DirectML", configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL
                };

                sessionOptions.AppendExecutionProvider_DML(deviceId);
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }



        public static OnnxExecutionProvider RyzenAI(int deviceId, PipelineType pipelineType)
        {
            var vaeKey = pipelineType == PipelineType.StableDiffusion3 ? "SD30_DECODER" : "SD15_DECODER";
            var transformerKey = pipelineType == PipelineType.StableDiffusion3 ? "SD30_MMDIT" : "SD15_UNET";

            return new OnnxExecutionProvider("RyzenAI", configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL
                };

                var variant = "RyzenAI";
                var variantModelPath = Path.Combine(Path.GetDirectoryName(configuration.OnnxModelPath), variant);
                if (Directory.Exists(variantModelPath))
                {
                    var modelCache = Path.Combine(variantModelPath, ".cache");
                    var dynamicDispatch = Path.Combine(Environment.CurrentDirectory, "RyzenAI");
                    var modelFolderPath = variantModelPath
                        .Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.TrimEntries)
                        .TakeLast(2)
                        .ToArray();

                    var modelName = modelFolderPath switch
                    {
                        var a when a.Contains("unet") => transformerKey,
                        var a when a.Contains("controlnet") => transformerKey,
                        var a when a.Contains("transformer") => transformerKey,
                        var a when a.Contains("vae_encoder") => vaeKey,
                        var a when a.Contains("vae_decoder") => vaeKey,
                        _ => string.Empty
                    };

                    if (!string.IsNullOrEmpty(modelName))
                    {
                        // Set new model path
                        configuration.OnnxModelPath = Path.Combine(variantModelPath, "model.onnx");

                        // Set NPU variables
                        sessionOptions.AddSessionConfigEntry("dd_root", dynamicDispatch);
                        sessionOptions.AddSessionConfigEntry("dd_cache", modelCache);
                        sessionOptions.AddSessionConfigEntry("onnx_custom_ops_const_key", modelCache);
                        sessionOptions.AddSessionConfigEntry("model_name", modelName);
                        sessionOptions.RegisterCustomOpLibrary("onnx_custom_ops.dll");
                    }
                }

                // Default: add DirectML and CPU providers
                sessionOptions.AppendExecutionProvider_DML(deviceId);
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }


        public static OnnxExecutionProvider AMDGPU(int deviceId)
        {
            var cacheDirectory = Path.Combine(Environment.CurrentDirectory, "AMDGPU");
            return new OnnxExecutionProvider("AMDGPU", configuration =>
            {

                if (!Directory.Exists(cacheDirectory))
                    Directory.CreateDirectory(cacheDirectory);

                var sessionOptions = new SessionOptions
                {
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL
                };

                sessionOptions.AppendExecutionProvider_MIGraphX(new OrtMIGraphXProviderOptions()
                {
                    DeviceId = deviceId,
                    ModelCacheDir = cacheDirectory
                });

                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }
    }
}
