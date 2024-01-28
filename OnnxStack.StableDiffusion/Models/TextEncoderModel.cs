using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class TextEncoderModel : OnnxModelSession
    {
        public TextEncoderModel(TextEncoderModelConfig configuration) : base(configuration)
        {

        }
    }

    public record TextEncoderModelConfig : OnnxModelConfig
    {

    }
}
