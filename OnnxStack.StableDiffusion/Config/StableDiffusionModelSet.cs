using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Enums;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text.Json.Serialization;

namespace OnnxStack.StableDiffusion.Config
{
    public class StableDiffusionModelSet : IOnnxModelSetConfig
    {
        public string Name { get; set; }
        public bool IsEnabled { get; set; }
        public int PadTokenId { get; set; }
        public int BlankTokenId { get; set; }
        public int TokenizerLimit { get; set; }
        public int TokenizerLength { get; set; }
        public int Tokenizer2Length { get; set; }
        public TokenizerType TokenizerType { get; set; }
        public int SampleSize { get; set; } = 512;
        public float ScaleFactor { get; set; }
        public ModelType ModelType { get; set; }
        public DiffuserPipelineType PipelineType { get; set; }
        public List<DiffuserType> Diffusers { get; set; } = new List<DiffuserType>();

        public int DeviceId { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }
        public List<OnnxModelConfig> ModelConfigurations { get; set; }

        [JsonIgnore]
        public ImmutableArray<int> BlankTokenValueArray { get; set; }

        public void InitBlankTokenArray()
        {
            BlankTokenValueArray = Enumerable.Repeat(BlankTokenId, 20480).ToImmutableArray();
        }
    }
}
