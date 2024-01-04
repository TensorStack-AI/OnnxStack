using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.Config
{
    public record ModelOptions(StableDiffusionModelSet BaseModel, ControlNetModelSet ControlNetModel = default)
    {
        public string Name => BaseModel.Name;
        public DiffuserPipelineType PipelineType => BaseModel.PipelineType;
        public float ScaleFactor => BaseModel.ScaleFactor;
        public ModelType ModelType => BaseModel.ModelType;
    }
}
