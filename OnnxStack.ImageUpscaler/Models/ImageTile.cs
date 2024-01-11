using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxStack.ImageUpscaler.Models
{
    public record ImageTile
    {
        public Image<Rgba32> Image { get; set; }
        public Rectangle Destination { get; set; }
    }
}



