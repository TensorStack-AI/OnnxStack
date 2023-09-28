using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Schedulers
{
    public interface IScheduler
    {
        float InitNoiseSigma { get; }
        IReadOnlyList<int> Timesteps { get; }
        DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep);
        DenseTensor<float> Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4);
    }
}