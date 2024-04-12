using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class AutoEncoderModel : OnnxModelSession
    {
        private readonly AutoEncoderModelConfig _configuration;

        public AutoEncoderModel(AutoEncoderModelConfig configuration) : base(configuration)
        {
            _configuration = configuration;
        }

        public float ScaleFactor => _configuration.ScaleFactor;
    }

    public record AutoEncoderModelConfig : OnnxModelConfig
    {
        public float ScaleFactor { get; set; }
    }
}
