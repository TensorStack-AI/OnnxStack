using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.Models
{
    public class UNetConditionModel : OnnxModelSession
    {
        private readonly UNetConditionModelConfig _configuration;

        public UNetConditionModel(UNetConditionModelConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
        }

        public ModelType ModelType => _configuration.ModelType;

        public int ContextSize => _configuration.ContextSize;
        public int FrameRate => _configuration.FrameRate;

        public static UNetConditionModel Create(UNetConditionModelConfig configuration)
        {
            return new UNetConditionModel(configuration);
        }

        public static UNetConditionModel Create(OnnxExecutionProvider executionProvider, string modelFile, ModelType modelType)
        {
            var configuration = new UNetConditionModelConfig
            {
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider,
                ModelType = modelType,
                ContextSize = 16,
                FrameRate = 8
            };
            return new UNetConditionModel(configuration);
        }
    }


    public record UNetConditionModelConfig : OnnxModelConfig
    {
        public ModelType ModelType { get; set; }
        public int ContextSize { get; set; }
        public int FrameRate { get; set; } = 8;
    }
}
