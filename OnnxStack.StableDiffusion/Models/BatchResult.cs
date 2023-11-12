using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.StableDiffusion.Models
{
    public record BatchResult(SchedulerOptions SchedulerOptions, DenseTensor<float> ImageResult);
}
