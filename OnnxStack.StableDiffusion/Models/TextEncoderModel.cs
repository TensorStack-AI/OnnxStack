using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class TextEncoderModel : OnnxModelSession
    {
        private readonly TextEncoderModelConfig _configuration;

        public TextEncoderModel(TextEncoderModelConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
        }

        public static TextEncoderModel Create(TextEncoderModelConfig configuration)
        {
            return new TextEncoderModel(configuration);
        }

        public static TextEncoderModel Create(OnnxExecutionProvider executionProvider, string modelFile)
        {
            var configuration = new TextEncoderModelConfig
            {
                OnnxModelPath = modelFile,
                ExecutionProvider = executionProvider
            };
            return new TextEncoderModel(configuration);
        }
    }

    public record TextEncoderModelConfig : OnnxModelConfig
    {

    }
}
