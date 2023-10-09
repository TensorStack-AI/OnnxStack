using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.IO;

namespace OnnxStack.StableDiffusion.Helpers
{
    internal static class ImageHelpers
    {
        /// <summary>
        /// Converts to image.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static Image<Rgb24> ToImage(this DenseTensor<float> imageTensor)
        {
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            var result = new Image<Rgb24>(width, height);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgb24(
                        CalculateByte(imageTensor, 0, y, x),
                        CalculateByte(imageTensor, 1, y, x),
                        CalculateByte(imageTensor, 2, y, x)
                    );
                }
            }
            return result;
        }


        /// <summary>
        /// Converts to image byte array.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static byte[] ToImageBytes(this DenseTensor<float> imageTensor)
        {
            using (var image = imageTensor.ToImage())
            using (var memoryStream = new MemoryStream())
            {
                image.SaveAsPng(memoryStream);
                return memoryStream.ToArray();
            }
        }


        /// <summary>
        /// Converts to image memory stream.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        public static Stream ToImageStream(this DenseTensor<float> imageTensor)
        {
            using (var image = imageTensor.ToImage())
            {
                var memoryStream = new MemoryStream();
                image.SaveAsPng(memoryStream);
                return memoryStream;
            }
        }


        public static DenseTensor<float> ToDenseTensor(this InputImage imageData, int width, int height)
        {
            if (!string.IsNullOrEmpty(imageData.ImagePath))
                return TensorFromFile(imageData.ImagePath, width, height);
            if(imageData.ImageBytes != null)
                return TensorFromBytes(imageData.ImageBytes, width, height);
            if (imageData.ImageStream != null)
                return TensorFromStream(imageData.ImageStream, width, height);
            if (imageData.ToDenseTensor != null)
                return imageData.ImageTensor.ToDenseTensor(); // Note: Tensor Copy

            return null;
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
        public static DenseTensor<float> TensorFromFile(string filename, int width, int height)
        {
            using (Image<Rgb24> image = Image.Load<Rgb24>(filename))
            {
                Resize(image, width, height);
                return ProcessPixels(width, height, image);
            }
        }


        public static DenseTensor<float> TensorFromBytes(byte[] imageBytes, int width, int height)
        {
            using (var image = Image.Load<Rgb24>(imageBytes))
            {
                Resize(image, width, height);
                return ProcessPixels(width, height, image);
            }
        }

        public static DenseTensor<float> TensorFromStream(Stream imageStream, int width, int height)
        {
            using (var image = Image.Load<Rgb24>(imageStream))
            {
                Resize(image, width, height);
                return ProcessPixels(width, height, image);
            }
        }




        private static DenseTensor<float> ProcessPixels(int width, int height, Image<Rgb24> image)
        {
            var imageArray = new DenseTensor<float>(new[] { 1, 3, width, height });
            image.ProcessPixelRows(img =>
            {
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        imageArray[0, 0, y, x] = (pixelSpan[x].R / 255.0f) * 2.0f - 1.0f;
                        imageArray[0, 1, y, x] = (pixelSpan[x].G / 255.0f) * 2.0f - 1.0f;
                        imageArray[0, 2, y, x] = (pixelSpan[x].B / 255.0f) * 2.0f - 1.0f;
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
        public static byte CalculateByte(Tensor<float> imageTensor, int index, int y, int x)
        {
            return (byte)Math.Round(Math.Clamp(imageTensor[0, index, y, x] / 2 + 0.5, 0, 1) * 255);
        }
    }
}
