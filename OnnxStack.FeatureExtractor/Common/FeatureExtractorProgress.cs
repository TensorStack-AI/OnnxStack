using OnnxStack.Core.Image;

namespace OnnxStack.FeatureExtractor.Common
{
    public record FeatureExtractorProgress(OnnxImage Source, OnnxImage Result, double Elapsed)
    {

    }
}
