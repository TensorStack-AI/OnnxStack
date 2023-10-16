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
        public static Image<Rgba32> ToImage(this DenseTensor<float> imageTensor)
        {
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            var hasAlpha = imageTensor.Dimensions[1] == 4;
            var result = new Image<Rgba32>(width, height);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgba32(
                        CalculateByte(imageTensor, 0, y, x),
                        CalculateByte(imageTensor, 1, y, x),
                        CalculateByte(imageTensor, 2, y, x),
                        hasAlpha ? CalculateByte(imageTensor, 3, y, x) : byte.MaxValue
                    );
                }
            }
            return result;
        }


        /// <summary>
        /// Converts to image.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        public static Image<Rgba32> ToImage(this InputImage inputImage)
        {
            if (!string.IsNullOrEmpty(inputImage.ImageBase64))
                return Image.Load<Rgba32>(Convert.FromBase64String(inputImage.ImageBase64.Split(',')[1]));
            if (inputImage.ImageBytes != null)
                return Image.Load<Rgba32>(inputImage.ImageBytes);
            if (inputImage.ImageStream != null)
                return Image.Load<Rgba32>(inputImage.ImageStream);
            if (inputImage.ToDenseTensor != null)
                return inputImage.ImageTensor.ToImage();

            return null;
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


        /// <summary>
        /// Converts to <see cref="DenseTensor<float>"/>.
        /// </summary>
        /// <param name="imageData">The image data.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static DenseTensor<float> ToDenseTensor(this InputImage imageData, ReadOnlySpan<int> dimensions)
        {
            if (!string.IsNullOrEmpty(imageData.ImageBase64))
                return TensorFromBase64(imageData.ImageBase64, dimensions);
            if (imageData.ImageBytes != null)
                return TensorFromBytes(imageData.ImageBytes, dimensions);
            if (imageData.ImageStream != null)
                return TensorFromStream(imageData.ImageStream, dimensions);
            if (imageData.ImageTensor != null)
                return imageData.ImageTensor.ToDenseTensor(); // Note: Tensor Copy // TODO: Reshape to dimensions

            return null;
        }


        /// <summary>
        /// Resizes the specified image.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        public static void Resize(Image image, ReadOnlySpan<int> dimensions)
        {
            var width = dimensions[3];
            var height = dimensions[2];
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
        public static void TensorToImageDebug(DenseTensor<float> imageTensor, string filename)
        {
            var width = imageTensor.Dimensions[3];
            var height = imageTensor.Dimensions[2];
            using (var result = new Image<Rgba32>(width, height))
            {
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
                result.SaveAsPng(filename);
            }
        }

        /// <summary>
        /// DEBUG ONLY: Convert a Tensor to image.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="size">The size.</param>
        /// <param name="filename">The filename.</param>
        public static void TensorToImageDebug2(DenseTensor<float> imageTensor, string filename)
        {
            var width = imageTensor.Dimensions[3];
            var height = imageTensor.Dimensions[2];
            using (var result = new Image<L8>(width, height))
            {
                for (var y = 0; y < height; y++)
                {
                    for (var x = 0; x < width; x++)
                    {
                        result[x, y] = new L8(CalculateByte(imageTensor, 0, y, x));
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
        public static DenseTensor<float> TensorFromFile(string filename, ReadOnlySpan<int> dimensions)
        {
            using (var image = Image.Load<Rgba32>(filename))
            {
                Resize(image, dimensions);
                return ProcessPixels(image, dimensions);
            }
        }


        /// <summary>
        /// Create an image Tensor from base64 string.
        /// </summary>
        /// <param name="base64Image">The base64 image.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public static DenseTensor<float> TensorFromBase64(string base64Image, ReadOnlySpan<int> dimensions)
        {
            using (var image = Image.Load<Rgba32>(Convert.FromBase64String(base64Image.Split(',')[1])))
            {
                Resize(image, dimensions);
                return ProcessPixels(image, dimensions);
            }
        }


        /// <summary>
        /// Create an image Tensor from bytes.
        /// </summary>
        /// <param name="imageBytes">The image bytes.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns>
        /// </returns>
        public static DenseTensor<float> TensorFromBytes(byte[] imageBytes, ReadOnlySpan<int> dimensions)
        {
            using (var image = Image.Load<Rgba32>(imageBytes))
            {
                Resize(image, dimensions);
                return ProcessPixels(image, dimensions);
            }
        }


        /// <summary>
        /// Create an image Tensor from stream.
        /// </summary>
        /// <param name="imageStream">The image stream.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns></returns>
        public static DenseTensor<float> TensorFromStream(Stream imageStream, ReadOnlySpan<int> dimensions)
        {
            using (var image = Image.Load<Rgba32>(imageStream))
            {
                Resize(image, dimensions);
                return ProcessPixels(image, dimensions);
            }
        }


        /// <summary>
        /// Processes the image pixels and places them into the tensor with the specified shape (batch, channels, height, width).
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        private static DenseTensor<float> ProcessPixels(Image<Rgba32> image, ReadOnlySpan<int> dimensions)
        {
            var width = dimensions[3];
            var height = dimensions[2];
            var channels = dimensions[1];
            var imageArray = new DenseTensor<float>(new[] { 1, channels, width, height });
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
                        if (channels == 4)
                            imageArray[0, 3, y, x] = (pixelSpan[x].A / 255.0f) * 2.0f - 1.0f;
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
