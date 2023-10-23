using OnnxStack.Core.Config;
using System.Collections.Immutable;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IModelOptions : IOnnxModel
    {
        int BlankTokenId { get; set; }
        ImmutableArray<int> BlankTokenValueArray { get; set; }
        int EmbeddingsLength { get; set; }
        int InputTokenLimit { get; set; }
        int PadTokenId { get; set; }
        float ScaleFactor { get; set; }
        int TokenizerLimit { get; set; }
    }
}