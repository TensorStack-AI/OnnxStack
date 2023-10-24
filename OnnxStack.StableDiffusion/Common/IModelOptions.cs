using OnnxStack.Core.Config;
using System.Collections.Immutable;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IModelOptions : IOnnxModel
    {
        int PadTokenId { get; set; }
        int BlankTokenId { get; set; }
        float ScaleFactor { get; set; }
        int TokenizerLimit { get; set; }
        int InputTokenLimit { get; set; }
        int EmbeddingsLength { get; set; }
        ImmutableArray<int> BlankTokenValueArray { get; set; }
    }
}