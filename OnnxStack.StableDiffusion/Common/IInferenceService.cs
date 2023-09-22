using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using System;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IInferenceService
    {
        int[] TokenizeText(string text);
        DenseTensor<float> PreprocessText(string prompt, string negativePrompt);
        Tensor<float> RunInference(StableDiffusionOptions options, SchedulerOptions schedulerOptions);
    }
}