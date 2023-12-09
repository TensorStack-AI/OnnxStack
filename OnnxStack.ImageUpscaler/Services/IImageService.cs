using OnnxStack.ImageUpscaler.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Generic;

namespace OnnxStack.ImageUpscaler.Services
{
    public interface IImageService
    {
        /// <summary>
        /// Generates the image tiles.
        /// </summary>
        /// <param name="imageSource">The image source.</param>
        /// <param name="sampleSize">Maximum size of the tile.</param>
        /// <param name="scaleFactor">The scale factor.</param>
        /// <returns></returns>
        List<ImageTile> GenerateTiles(Image<Rgba32> imageSource, int sampleSize, int scaleFactor);
    }
}