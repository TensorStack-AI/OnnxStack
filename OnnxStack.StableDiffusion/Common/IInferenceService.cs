using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using System;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IInferenceService : IDisposable
    {
        int[] TokenizeText(string text);
        DenseTensor<float> PreprocessText(string prompt, string negativePrompt);
        Tensor<float> RunInference(StableDiffusionOptions options, SchedulerOptions schedulerOptions);
    }
}