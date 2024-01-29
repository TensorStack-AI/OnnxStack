using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class ControlNetModel : OnnxModelSession
    {
        public ControlNetModel(ControlNetModelConfig configuration)
            : base(configuration) { }

        public static ControlNetModel Create(ControlNetModelConfig configuration)
        {
            return new ControlNetModel(configuration);
        }

        public static ControlNetModel Create(string modelFile, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new ControlNetModelConfig
            {
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelFile
            };
            return new ControlNetModel(configuration);
        }
    }

    public record ControlNetModelConfig : OnnxModelConfig;
}
