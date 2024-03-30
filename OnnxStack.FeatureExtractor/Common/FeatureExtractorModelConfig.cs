using OnnxStack.Core.Config;
using OnnxStack.Core.Image;

namespace OnnxStack.FeatureExtractor.Common
{
    public record FeatureExtractorModelConfig : OnnxModelConfig
    {
        public int SampleSize { get; set; }
        public int OutputChannels { get; set; }
        public bool NormalizeOutputTensor { get; set; }
        public bool SetOutputToInputAlpha { get; set; }
        public ImageResizeMode InputResizeMode { get; set; }
        public ImageNormalizeType NormalizeInputTensor { get; set; }
    }
}
