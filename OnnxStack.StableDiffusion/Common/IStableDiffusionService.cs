using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IStableDiffusionService : IDisposable
    {
        Task<Image<Rgba32>> TextToImage(StableDiffusionOptions options);
        Task<Image<Rgba32>> TextToImage(StableDiffusionOptions options, SchedulerOptions schedulerOptions);

        Task<bool> TextToImageFile(StableDiffusionOptions options, string outputFile);
        Task<bool> TextToImageFile(StableDiffusionOptions options, SchedulerOptions schedulerOptions, string outputFile);
    }
}