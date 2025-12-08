using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Tokenizers;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace OnnxStack.StableDiffusion.Config
{
    public record StableDiffusionModelSet : IOnnxModelSetConfig
    {
        public string Name { get; set; }
        public int SampleSize { get; set; } = 512;
        public PipelineType PipelineType { get; set; }
        public List<DiffuserType> Diffusers { get; set; }
        public OnnxExecutionProvider ExecutionProvider { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public List<SchedulerType> Schedulers { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TokenizerConfig TokenizerConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TokenizerConfig Tokenizer2Config { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TokenizerConfig Tokenizer3Config { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TextEncoderModelConfig TextEncoderConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TextEncoderModelConfig TextEncoder2Config { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public TextEncoderModelConfig TextEncoder3Config { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public UNetConditionModelConfig UnetConfig { get; set; }
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public UNetConditionModelConfig Unet2Config { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public AutoEncoderModelConfig VaeDecoderConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public AutoEncoderModelConfig VaeEncoderConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public UNetConditionModelConfig ControlNetUnetConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public FlowEstimationModelConfig FlowEstimationConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public ResampleModelConfig ResampleModelConfig { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public SchedulerOptions SchedulerOptions { get; set; }
    }
}
