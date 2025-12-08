using OnnxStack.Core.Config;
using OnnxStack.Core.Image;

namespace OnnxStack.FeatureExtractor.Common
{
    public record FeatureExtractorModelConfig : OnnxModelConfig
    {
        public string Name { get; set; }
        public int SampleSize { get; set; }
        public int OutputChannels { get; set; }
        public ImageNormalizeType NormalizeType { get; set; }
        public ImageNormalizeType NormalizeOutputType { get; set; }
        public bool SetOutputToInputAlpha { get; set; }
        public ImageResizeMode InputResizeMode { get; set; }
        public bool InvertOutput { get; set; }
    }
}
