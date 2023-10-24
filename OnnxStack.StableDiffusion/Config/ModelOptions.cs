using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Common;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace OnnxStack.StableDiffusion.Config
{
    public class ModelOptions : IModelOptions, IOnnxModelSetConfig
    {
        public string Name { get; set; }
        public int PadTokenId { get; set; }
        public int BlankTokenId { get; set; }
        public int InputTokenLimit { get; set; }
        public int TokenizerLimit { get; set; }
        public int EmbeddingsLength { get; set; }
        public float ScaleFactor { get; set; }
        public ImmutableArray<int> BlankTokenValueArray { get; set; }

        public int DeviceId { get; set; }
        public string OnnxModelPath { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }
        public List<OnnxModelSessionConfig> ModelConfigurations { get; set; }
    }
}
