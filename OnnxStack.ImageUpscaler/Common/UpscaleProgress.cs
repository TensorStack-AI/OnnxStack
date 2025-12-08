using OnnxStack.Core.Image;

namespace OnnxStack.ImageUpscaler.Common
{
    public record UpscaleProgress(OnnxImage Source, OnnxImage Result, double Elapsed)
    {

    }
}
