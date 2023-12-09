using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;

namespace OnnxStack.ImageUpscaler.Extensions
{
    internal static class ImageExtensions
    {

        /// <summary>
        /// Converts to DenseTensor.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> ToDenseTensor(this Image<Rgba32> image, ReadOnlySpan<int> dimensions)
        {
            using (image)
            {
                return ProcessPixels(image, dimensions);
            }
        }


        /// <summary>
        /// Converts to Image.
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns></returns>
        public static Image<Rgba32> ToImage(this OrtValue ortValue)
        {
            return ortValue.ToDenseTensor().ToImage();
        }


        /// <summary>
        /// Converts to image.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static Image<Rgba32> ToImage(this DenseTensor<float> imageTensor)
        {
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            var result = new Image<Rgba32>(width, height);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
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


        /// <summary>
        /// Processes the pixels.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        private static DenseTensor<float> ProcessPixels(Image<Rgba32> image, ReadOnlySpan<int> dimensions)
        {
            var width = dimensions[3];
            var height = dimensions[2];
            var channels = dimensions[1];
            var imageArray = new DenseTensor<float>(new[] { 1, channels, height, width });
            image.ProcessPixelRows(img =>
            {
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        imageArray[0, 0, y, x] = (pixelSpan[x].R / 255.0f);
                        imageArray[0, 1, y, x] = (pixelSpan[x].G / 255.0f);
                        imageArray[0, 2, y, x] = (pixelSpan[x].B / 255.0f);
                    }
                }
            });
            return imageArray;
        }


        /// <summary>
        /// Calculates the byte.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="index">The index.</param>
        /// <param name="y">The y.</param>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        private static byte CalculateByte(Tensor<float> imageTensor, int index, int y, int x)
        {
            return (byte)Math.Round(Math.Clamp(imageTensor[0, index, y, x], 0, 1) * 255);
        }
    }
}
