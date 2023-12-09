using OnnxStack.ImageUpscaler.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Generic;

namespace OnnxStack.ImageUpscaler.Services
{
    public interface IImageService
    {
        List<ImageTile> GenerateTiles(Image<Rgba32> imageSource, int maxTileSize, int scaleFactor);
    }
}