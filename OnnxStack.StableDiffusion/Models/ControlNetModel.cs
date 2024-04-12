using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.Models
{
    public class ControlNetModel : OnnxModelSession
    {
        private readonly ControlNetModelConfig _configuration;
        public ControlNetModel(ControlNetModelConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
        }

        public ControlNetType Type => _configuration.Type;
        public DiffuserPipelineType PipelineType => _configuration.PipelineType;

        public static ControlNetModel Create(ControlNetModelConfig configuration)
        {
            return new ControlNetModel(configuration);
        }

        public static ControlNetModel Create(string modelFile, ControlNetType type, DiffuserPipelineType pipeline, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new ControlNetModelConfig
            {
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelFile,
                Type = type,
                PipelineType = pipeline,
            };
            return new ControlNetModel(configuration);
        }
    }

    public record ControlNetModelConfig : OnnxModelConfig
    {
        public ControlNetType Type { get; set; }
        public DiffuserPipelineType PipelineType { get; set; }
    }
}
