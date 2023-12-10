using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxStack.ImageUpscaler.Models
{
    public record ImageTile(Image<Rgba32> Image, Rectangle Destination);
}



