using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IImageService
    {
        Image<Rgba32> TensorToImage(Tensor<float> imageTensor, int width, int height);
    }
}