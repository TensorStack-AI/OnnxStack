using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Results;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IStableDiffusionService
    {
        Task<ImageResult> TextToImage(PromptOptions prompt);
        Task<ImageResult> TextToImage(PromptOptions prompt, SchedulerOptions options);
        Task<ImageResult> TextToImage(PromptOptions prompt, SchedulerOptions options, Action<int, int> progress = null, CancellationToken cancellationToken = default);

        Task<ImageResult> TextToImageFile(PromptOptions prompt, string outputFile);
        Task<ImageResult> TextToImageFile(PromptOptions prompt, SchedulerOptions options, string outputFile);
        Task<ImageResult> TextToImageFile(PromptOptions prompt, SchedulerOptions options, string outputFile, Action<int, int> progress = null, CancellationToken cancellationToken = default);
    }
}