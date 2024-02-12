using OnnxStack.Core.Image;
using SixLabors.ImageSharp;

namespace OnnxStack.ImageUpscaler.Models
{
    public record ImageTile
    {
        public OnnxImage Image { get; set; }
        public Rectangle Destination { get; set; }
    }
}



