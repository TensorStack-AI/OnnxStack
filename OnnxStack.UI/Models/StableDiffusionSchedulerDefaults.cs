using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.UI.Models
{
    public record StableDiffusionSchedulerDefaults(
        SchedulerType SchedulerType = SchedulerType.EulerAncestral,
        int Steps = 30, int StepsMin = 4, int StepsMax = 100,
        float Guidance = 7.5f, float GuidanceMin = 0f, float GuidanceMax = 30f);
}
