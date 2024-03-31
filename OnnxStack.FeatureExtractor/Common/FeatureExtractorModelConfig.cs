using OnnxStack.Core.Config;
using OnnxStack.Core.Image;

namespace OnnxStack.FeatureExtractor.Common
{
    public record FeatureExtractorModelConfig : OnnxModelConfig
    {
        public int SampleSize { get; set; }
        public int OutputChannels { get; set; }
        public bool NormalizeOutput { get; set; }
        public bool SetOutputToInputAlpha { get; set; }
        public ImageResizeMode InputResizeMode { get; set; }
        public ImageNormalizeType NormalizeType { get; set; }
        public bool NormalizeInput { get; set; }
    }
}
