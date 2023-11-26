using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Enums;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IModelOptions : IOnnxModel
    {
        bool IsEnabled { get; set; }
        int PadTokenId { get; set; }
        int BlankTokenId { get; set; }
        int SampleSize { get; set; }
        float ScaleFactor { get; set; }
        int TokenizerLimit { get; set; }
        int EmbeddingsLength { get; set; }
        int DualEmbeddingsLength { get; set; }
        bool IsDualTokenizer { get; set; }
        DiffuserPipelineType PipelineType { get; set; }
        List<DiffuserType> Diffusers { get; set; }
        ImmutableArray<int> BlankTokenValueArray { get; set; }
    }
}