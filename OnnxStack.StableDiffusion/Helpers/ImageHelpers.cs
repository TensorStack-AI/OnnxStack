using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Results;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
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
        /// Converts an DenseTensor image to Image<Rgba32>
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns></returns>
        public static Image<Rgba32> TensorToImage(DenseTensor<float> imageTensor, int width, int height)
        {
            var image = new Image<Rgba32>(width, height);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    image[x, y] = new Rgba32(
                        CalculateByte(imageTensor, 0, y, x),
                        CalculateByte(imageTensor, 1, y, x),
                        CalculateByte(imageTensor, 2, y, x)
                    );
                }
            }
            return image;
        }


        /// <summary>
        /// Resizes the specified image.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns></returns>
        public static void Resize(Image image, int width, int height)
        {
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(width, height),
                    Mode = ResizeMode.Crop
                });
            });
        }


        /// <summary>
        /// DEBUG ONLY: Convert a Tensor to image.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="size">The size.</param>
        /// <param name="filename">The filename.</param>
        public static void TensorToImageDebug(DenseTensor<float> imageTensor, int size, string filename)
        {
            using (var result = new Image<Rgba32>(size, size))
            {
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
        }


        /// <summary>
        /// Loads and image from file and loads it iton a DenseTesor with the shape 1, 3, W, H.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns></returns>
        public static DenseTensor<float> TensorFromImage(string filename, int width, int height)
        {
            using (Image<Rgb24> image = Image.Load<Rgb24>(filename))
            {
                Resize(image, width, height);
                var imageArray = new DenseTensor<float>(new[] { 1, 3, width, height });
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        var pixelSpan = image.GetPixelRowSpan(y);
                        imageArray[0, 0, y, x] = (pixelSpan[x].R / 255.0f) * 2.0f - 1.0f;
                        imageArray[0, 1, y, x] = (pixelSpan[x].G / 255.0f) * 2.0f - 1.0f;
                        imageArray[0, 2, y, x] = (pixelSpan[x].B / 255.0f) * 2.0f - 1.0f;
                    }
                }
                return imageArray;
            }
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
