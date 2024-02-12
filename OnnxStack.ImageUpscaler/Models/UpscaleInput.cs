using System.Collections.Generic;

namespace OnnxStack.ImageUpscaler.Models
{
    internal record UpscaleInput(List<ImageTile> ImageTiles, int OutputWidth, int OutputHeight);
}



