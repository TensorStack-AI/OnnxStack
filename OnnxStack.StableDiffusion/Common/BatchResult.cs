using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.StableDiffusion.Common
{
    public record BatchResult(SchedulerOptions SchedulerOptions, OnnxImage ImageResult);
}
