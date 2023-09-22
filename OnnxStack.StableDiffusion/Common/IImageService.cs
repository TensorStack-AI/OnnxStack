using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IImageService
    {
        Image<Rgba32> TensorToImage(StableDiffusionOptions options, Tensor<float> imageTensor);
    }
}