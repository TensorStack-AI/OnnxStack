using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class TextEncoderModel : OnnxModelSession
    {
        private readonly TextEncoderModelConfig _configuration;

        public TextEncoderModel(TextEncoderModelConfig configuration) : base(configuration)
        {
            _configuration = configuration;
        }
    }

    public record TextEncoderModelConfig : OnnxModelConfig
    {

    }
}
