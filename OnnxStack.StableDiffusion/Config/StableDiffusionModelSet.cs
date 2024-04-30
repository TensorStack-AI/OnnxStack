using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace OnnxStack.StableDiffusion.Config
{
    public record StableDiffusionModelSet : IOnnxModelSetConfig
    {
        public string Name { get; set; }
        public bool IsEnabled { get; set; }
        public int SampleSize { get; set; } = 512;
        public DiffuserPipelineType PipelineType { get; set; }
        public List<DiffuserType> Diffusers { get; set; } = new List<DiffuserType>();
        public MemoryModeType MemoryMode { get; set; }
        public int DeviceId { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }
        public OnnxModelPrecision Precision { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TokenizerModelConfig TokenizerConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TokenizerModelConfig Tokenizer2Config { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TextEncoderModelConfig TextEncoderConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TextEncoderModelConfig TextEncoder2Config { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public UNetConditionModelConfig UnetConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public AutoEncoderModelConfig VaeDecoderConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public AutoEncoderModelConfig VaeEncoderConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public UNetConditionModelConfig DecoderUnetConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public SchedulerOptions SchedulerOptions { get; set; }

      
    }
}
