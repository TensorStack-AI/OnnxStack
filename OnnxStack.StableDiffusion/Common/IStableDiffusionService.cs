using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IStableDiffusionService : IDisposable
    {
        Task<Image<Rgba32>> TextToImage(string prompt);
        Task<Image<Rgba32>> TextToImage(string prompt, string negativePrompt);
        Task<Image<Rgba32>> TextToImage(string prompt, SchedulerConfig schedulerConfig);
        Task<Image<Rgba32>> TextToImage(string prompt, string negativePrompt, SchedulerConfig schedulerConfig);

        Task<bool> TextToImageFile(string prompt, string filename);
        Task<bool> TextToImageFile(string prompt, string negativePrompt, string filename);
        Task<bool> TextToImageFile(string prompt, string filename, SchedulerConfig schedulerConfig);
        Task<bool> TextToImageFile(string prompt, string negativePrompt, string filename, SchedulerConfig schedulerConfig);
    }
}