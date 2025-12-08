using OnnxStack.Core.Config;
using OnnxStack.Core.Image;
using System.IO;

namespace OnnxStack.ImageUpscaler.Common
{
    public record UpscaleModelConfig : OnnxModelConfig
    {
        public string Name { get; set; }
        public int Channels { get; set; }
        public int SampleSize { get; set; }
        public int ScaleFactor { get; set; }
        public ImageNormalizeType NormalizeType { get; set; }
        public TileMode TileMode { get; set; }
        public int TileSize { get; set; }
        public int TileOverlap { get; set; }
    }
}
