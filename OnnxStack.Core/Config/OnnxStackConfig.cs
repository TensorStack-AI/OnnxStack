using OnnxStack.Common.Config;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace OnnxStack.Core.Config
{
    public class OnnxStackConfig : IConfigSection
    {
        public string Name { get; set; }
        public int PadTokenId { get; set; }
        public int BlankTokenId { get; set; }
        public int InputTokenLimit { get; set; }
        public int TokenizerLimit { get; set; }
        public int EmbeddingsLength { get; set; }
        public float ScaleFactor { get; set; }
        public List<OnnxModelSessionConfig> ModelConfigurations { get; set; }
        public  ImmutableArray<int> BlankTokenValueArray { get; set; }

        public void Initialize()
        {
            BlankTokenValueArray = Enumerable.Repeat(BlankTokenId, InputTokenLimit).ToImmutableArray();
        }
    }
}
