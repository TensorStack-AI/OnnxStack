using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.UI.Models
{
    public record ModelOptions(StableDiffusionModelSet BaseModel, ControlNetModelSet ControlNetModel = default);
}
