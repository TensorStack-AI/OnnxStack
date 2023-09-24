using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Results;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IStableDiffusionService
    {
        Task<ImageResult> TextToImage(StableDiffusionOptions options);
        Task<ImageResult> TextToImage(StableDiffusionOptions options, SchedulerOptions schedulerOptions);

        Task<ImageResult> TextToImageFile(StableDiffusionOptions options, string outputFile);
        Task<ImageResult> TextToImageFile(StableDiffusionOptions options, SchedulerOptions schedulerOptions, string outputFile);
    }
}