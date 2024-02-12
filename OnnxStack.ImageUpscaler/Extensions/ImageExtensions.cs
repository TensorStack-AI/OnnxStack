using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using OnnxStack.ImageUpscaler.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

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
        internal static List<ImageTile> GenerateTiles(this OnnxImage imageSource, int sampleSize, int scaleFactor)
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
                    tiles.Add(new ImageTile { Image = tileImage, Destination = tileDest });
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
        internal static OnnxImage ExtractTile(this OnnxImage sourceImage, Rectangle sourceArea)
        {
            var height = sourceArea.Height;
            var targetImage = new Image<Rgba32>(sourceArea.Width, sourceArea.Height);
            sourceImage.GetImage().ProcessPixelRows(targetImage, (sourceAccessor, targetAccessor) =>
            {
                for (int i = 0; i < height; i++)
                {
                    var sourceRow = sourceAccessor.GetRowSpan(sourceArea.Y + i);
                    var targetRow = targetAccessor.GetRowSpan(i);
                    sourceRow.Slice(sourceArea.X, sourceArea.Width).CopyTo(targetRow);
                }
            });
            return new OnnxImage(targetImage);
        }


        internal static void ApplyImageTile(this DenseTensor<float> imageTensor, DenseTensor<float> tileTensor, Rectangle location)
        {
            var offsetY = location.Y;
            var offsetX = location.X;
            var dimensions = tileTensor.Dimensions.ToArray();
            Parallel.For(0, dimensions[0], (i) =>
            {
                Parallel.For(0, dimensions[1], (j) =>
                {
                    Parallel.For(0, dimensions[2], (k) =>
                    {
                        Parallel.For(0, dimensions[3], (l) =>
                        {
                            imageTensor[i, j, k + offsetY, l + offsetX] = tileTensor[i, j, k, l];
                        });
                    });
                });
            });
        }
    }
}
