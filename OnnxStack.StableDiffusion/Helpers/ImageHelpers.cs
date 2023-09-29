using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Results;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;

namespace OnnxStack.StableDiffusion.Helpers
{
    internal static class ImageHelpers
    {
        /// <summary>
        /// Convert a Tensor to image.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static ImageResult TensorToImage(SchedulerOptions options, DenseTensor<float> imageTensor)
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
            return new ImageResult(result);
        }


        /// <summary>
        /// DEBUG ONLY: Convert a Tensor to image.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="size">The size.</param>
        /// <param name="filename">The filename.</param>
        public static void TensorToImageDebug(DenseTensor<float> imageTensor, int size, string filename)
        {
            var result = new Image<Rgba32>(size, size);
            for (var y = 0; y < size; y++)
            {
                for (var x = 0; x < size; x++)
                {
                    result[x, y] = new Rgba32(
                        CalculateByte(imageTensor, 0, y, x),
                        CalculateByte(imageTensor, 1, y, x),
                        CalculateByte(imageTensor, 2, y, x)
                    );
                }
            }
            result.SaveAsPng(filename);
        }


        /// <summary>
        /// Calculates the byte.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="index">The index.</param>
        /// <param name="y">The y.</param>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        public static byte CalculateByte(Tensor<float> imageTensor, int index, int y, int x)
        {
            return (byte)Math.Round(Math.Clamp(imageTensor[0, index, y, x] / 2 + 0.5, 0, 1) * 255);
        }
    }
}
