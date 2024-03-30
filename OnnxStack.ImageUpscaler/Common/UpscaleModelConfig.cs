using OnnxStack.Core.Config;

namespace OnnxStack.ImageUpscaler.Common
{
    public record UpscaleModelConfig : OnnxModelConfig
    {
        public int Channels { get; set; }
        public int SampleSize { get; set; }
        public int ScaleFactor { get; set; }

        public int TileSize { get; set; }
        public int TileOverlap { get; set; }
    }
}
