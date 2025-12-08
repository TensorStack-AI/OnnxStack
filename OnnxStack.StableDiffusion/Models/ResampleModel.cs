using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class ResampleModel : OnnxModelSession
    {
        private readonly ResampleModelConfig _configuration;

        public ResampleModel(ResampleModelConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
        }

        public static ResampleModel Create(ResampleModelConfig configuration)
        {
            return new ResampleModel(configuration);
        }

        public static ResampleModel Create(OnnxExecutionProvider executionProvider, string modelFile)
        {
            var configuration = new ResampleModelConfig
            {
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider
            };
            return new ResampleModel(configuration);
        }
    }

    public record ResampleModelConfig : OnnxModelConfig
    {
    }
}
