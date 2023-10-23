using OnnxStack.Common.Config;
using OnnxStack.StableDiffusion.Common;
using System.Collections.Immutable;
using System.Linq;

namespace OnnxStack.StableDiffusion.Config
{
    public class ModelOptions : IModelOptions
    {
        public string Name { get; set; }
        public int PadTokenId { get; set; }
        public int BlankTokenId { get; set; }
        public int InputTokenLimit { get; set; }
        public int TokenizerLimit { get; set; }
        public int EmbeddingsLength { get; set; }
        public float ScaleFactor { get; set; }
        public ImmutableArray<int> BlankTokenValueArray { get; set; }
    }
}
