using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.StableDiffusion.Common
{
    public record BatchResult(SchedulerOptions SchedulerOptions, DenseTensor<float> Result);
    public record BatchImageResult(SchedulerOptions SchedulerOptions, OnnxImage Result);
}
