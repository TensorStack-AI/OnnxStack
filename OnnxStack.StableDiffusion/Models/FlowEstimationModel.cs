using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class FlowEstimationModel : OnnxModelSession
    {
        private readonly FlowEstimationModelConfig _configuration;

        public FlowEstimationModel(FlowEstimationModelConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
        }

        public static FlowEstimationModel Create(FlowEstimationModelConfig configuration)
        {
            return new FlowEstimationModel(configuration);
        }

        public static FlowEstimationModel Create(OnnxExecutionProvider executionProvider, string modelFile)
        {
            var configuration = new FlowEstimationModelConfig
            {
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider
            };
            return new FlowEstimationModel(configuration);
        }
    }

    public record FlowEstimationModelConfig : OnnxModelConfig
    {
    }
}
