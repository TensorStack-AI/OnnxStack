using OnnxStack.ImageUpscaler.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;

namespace OnnxStack.ImageUpscaler.Services
{
    internal class ImageService : IImageService
    {
        /// <summary>
        /// Generates the image tiles.
        /// </summary>
        /// <param name="imageSource">The image source.</param>
        /// <param name="sampleSize">Maximum size of the tile.</param>
        /// <param name="scaleFactor">The scale factor.</param>
        /// <returns></returns>
        public List<ImageTile> GenerateTiles(Image<Rgba32> imageSource, int sampleSize, int scaleFactor)
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
        private static Image<Rgba32> ExtractTile(Image<Rgba32> sourceImage, Rectangle sourceArea)
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
    }
}
