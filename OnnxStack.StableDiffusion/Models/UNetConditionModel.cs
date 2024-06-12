using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.Models
{
    public class UNetConditionModel : OnnxModelSession
    {
        private readonly UNetConditionModelConfig _configuration;

        public UNetConditionModel(UNetConditionModelConfig configuration) : base(configuration)
        {
            _configuration = configuration;
        }

        public ModelType ModelType => _configuration.ModelType;

        public static UNetConditionModel Create(UNetConditionModelConfig configuration)
        {
            return new UNetConditionModel(configuration);
        }

        public static UNetConditionModel Create(string modelFile, ModelType modelType, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new UNetConditionModelConfig
            {
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelFile,
                ModelType = modelType
            };
            return new UNetConditionModel(configuration);
        }
    }


    public record UNetConditionModelConfig : OnnxModelConfig
    {
        public ModelType ModelType { get; set; }
    }
}
