using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class AutoEncoderModel : OnnxModelSession
    {
        private readonly float _scaleFactor;
        public AutoEncoderModel(AutoEncoderModelConfig configuration) : base(configuration)
        {
            _scaleFactor = configuration.ScaleFactor;
        }

        public float ScaleFactor => _scaleFactor;
    }

    public record AutoEncoderModelConfig : OnnxModelConfig
    {
        public float ScaleFactor { get; set; }
    }
}
