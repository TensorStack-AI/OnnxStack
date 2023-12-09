using OnnxStack.Core.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Threading.Tasks;

namespace OnnxStack.ImageUpscaler.Services
{
    public interface IUpscaleService
    {
        Task<Image<Rgba32>> GenerateAsync(IOnnxModel modelOptions, Image<Rgba32> inputImage);
    }
}
