using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.Config
{
    public record ModelOptions(StableDiffusionModelSet BaseModel, ControlNetModelSet ControlNetModel = default);
  
    public record ModelConfiguration(string Name, ModelType ModelType, DiffuserPipelineType PipelineType, int SampleSize);

}
