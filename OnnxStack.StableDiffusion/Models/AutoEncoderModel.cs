using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class AutoEncoderModel : OnnxModelSession
    {
        private readonly AutoEncoderModelConfig _configuration;

        public AutoEncoderModel(AutoEncoderModelConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
        }

        public float ScaleFactor => _configuration.ScaleFactor;


        public static AutoEncoderModel Create(AutoEncoderModelConfig configuration)
        {
            return new AutoEncoderModel(configuration);
        }

        public static AutoEncoderModel Create(OnnxExecutionProvider executionProvider, string modelFile, float scaleFactor)
        {
            var configuration = new AutoEncoderModelConfig
            {
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider,
                ScaleFactor = scaleFactor
            };
            return new AutoEncoderModel(configuration);
        }
    }

    public record AutoEncoderModelConfig : OnnxModelConfig
    {
        public float ScaleFactor { get; set; }
    }
}
