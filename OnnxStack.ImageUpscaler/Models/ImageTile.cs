using OnnxStack.Core.Image;
using SixLabors.ImageSharp;

namespace OnnxStack.ImageUpscaler.Models
{
    internal record ImageTile
    {
        public OnnxImage Image { get; set; }
        public Rectangle Destination { get; set; }
    }
}



