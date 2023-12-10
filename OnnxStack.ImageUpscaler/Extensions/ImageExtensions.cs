using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.ImageUpscaler.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;

namespace OnnxStack.ImageUpscaler.Extensions
{
    internal static class ImageExtensions
    {
        /// <summary>
        /// Generates the image tiles.
        /// </summary>
        /// <param name="imageSource">The image source.</param>
        /// <param name="sampleSize">Maximum size of the tile.</param>
        /// <param name="scaleFactor">The scale factor.</param>
        /// <returns></returns>
        public static List<ImageTile> GenerateTiles(this Image<Rgba32> imageSource, int sampleSize, int scaleFactor)
        {
            var tiles = new List<ImageTile>();
            var tileSizeX = Math.Min(sampleSize, imageSource.Width);
            var tileSizeY = Math.Min(sampleSize, imageSource.Height);
            var columns = (int)Math.Ceiling((double)imageSource.Width / tileSizeX);
            var rows = (int)Math.Ceiling((double)imageSource.Height / tileSizeY);
            var tileWidth = imageSource.Width / columns;
            var tileHeight = imageSource.Height / rows;

            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < columns; x++)
                {
                    var tileRect = new Rectangle(x * tileWidth, y * tileHeight, tileWidth, tileHeight);
                    var tileDest = new Rectangle(tileRect.X * scaleFactor, tileRect.Y * scaleFactor, tileWidth * scaleFactor, tileHeight * scaleFactor);
                    var tileImage = ExtractTile(imageSource, tileRect);
                    tiles.Add(new ImageTile(tileImage, tileDest));
                }
            }
            return tiles;
        }


        /// <summary>
        /// Extracts an image tile from a source image.
        /// </summary>
        /// <param name="sourceImage">The source image.</param>
        /// <param name="sourceArea">The source area.</param>
        /// <returns></returns>
        public static Image<Rgba32> ExtractTile(this Image<Rgba32> sourceImage, Rectangle sourceArea)
        {
            var height = sourceArea.Height;
            var targetImage = new Image<Rgba32>(sourceArea.Width, sourceArea.Height);
            sourceImage.ProcessPixelRows(targetImage, (sourceAccessor, targetAccessor) =>
            {
                for (int i = 0; i < height; i++)
                {
                    var sourceRow = sourceAccessor.GetRowSpan(sourceArea.Y + i);
                    var targetRow = targetAccessor.GetRowSpan(i);
                    sourceRow.Slice(sourceArea.X, sourceArea.Width).CopyTo(targetRow);
                }
            });
            return targetImage;
        }


        /// <summary>
        /// Converts to InputImage to an Image<Rgba32>
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
            if (inputImage.ImageTensor != null)
                return inputImage.ImageTensor.ToImage();

            return inputImage.Image;
        }

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
                return NormalizeToZeroToOne(image, dimensions);
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
                        DenormalizeZeroToOneToByte(imageTensor, 0, y, x),
                        DenormalizeZeroToOneToByte(imageTensor, 1, y, x),
                        DenormalizeZeroToOneToByte(imageTensor, 2, y, x)
                    );
                }
            }
            return result;
        }



        /// <summary>
        /// Normalizes the pixels from 0-255 to 0-1
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        private static DenseTensor<float> NormalizeToZeroToOne(Image<Rgba32> image, ReadOnlySpan<int> dimensions)
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
        /// Denormalizes the pixels from 0-1 to 0-255
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="index">The index.</param>
        /// <param name="y">The y.</param>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        private static byte DenormalizeZeroToOneToByte(DenseTensor<float> imageTensor, int index, int y, int x)
        {
            return (byte)Math.Round(Math.Clamp(imageTensor[0, index, y, x], 0, 1) * 255);
        }
    }
}
