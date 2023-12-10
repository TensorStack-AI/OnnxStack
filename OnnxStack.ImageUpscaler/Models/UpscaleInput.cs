using System.Collections.Generic;

namespace OnnxStack.ImageUpscaler.Models
{
    public record UpscaleInput(List<ImageTile> ImageTiles, int OutputWidth, int OutputHeight);
}



