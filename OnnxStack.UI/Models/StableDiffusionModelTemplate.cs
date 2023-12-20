using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.UI.Models
{
    public record StableDiffusionModelTemplate(DiffuserPipelineType PipelineType, ModelType ModelType, int SampleSize, DiffuserType[] DiffuserTypes)
    {
        public StableDiffusionSchedulerDefaults SchedulerDefaults { get; set; } = new StableDiffusionSchedulerDefaults();
    }
}
