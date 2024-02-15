using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.StableDiffusion.Common
{
    public record BatchResult(SchedulerOptions SchedulerOptions, DenseTensor<float> Result);
}
