using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.UI.Models
{
    public record StableDiffusionModelTemplate(string Name, DiffuserPipelineType PipelineType, ModelType ModelType, int SampleSize, params DiffuserType[] DiffuserTypes);
}
