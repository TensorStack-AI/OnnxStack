using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class ControlNetModel : OnnxModelSession
    {
        public ControlNetModel(ControlNetModelConfig configuration) : base(configuration)
        {

        }
    }

    public record ControlNetModelConfig : OnnxModelConfig
    {

    }
}
