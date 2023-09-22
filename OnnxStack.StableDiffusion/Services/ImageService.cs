using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;

namespace OnnxStack.StableDiffusion.Services
{
    public class ImageService : IImageService
    {
        public Image<Rgba32> TensorToImage(StableDiffusionOptions options, Tensor<float> imageTensor)
        {
            var result = new Image<Rgba32>(options.Width, options.Height);
            for (var y = 0; y < options.Height; y++)
            {
                for (var x = 0; x < options.Width; x++)
                {
                    result[x, y] = new Rgba32(
                        CalculateByte(imageTensor, 0, y, x),
                        CalculateByte(imageTensor, 1, y, x),
                        CalculateByte(imageTensor, 2, y, x)
                    );
                }
            }
            return result;
        }

        private static byte CalculateByte(Tensor<float> imageTensor, int index, int y, int x)
        {
            return (byte)Math.Round(Math.Clamp(imageTensor[0, index, y, x] / 2 + 0.5, 0, 1) * 255);
        }
    }
}
