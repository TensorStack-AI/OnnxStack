using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System.Text.Json.Serialization;

namespace OnnxStack.StableDiffusion.Config
{
    public record ControlNetModelSet : IOnnxModelSetConfig
    {
        public ControlNetType Type { get; set; }
        public DiffuserPipelineType PipelineType { get; set; }
        public string Name { get; set; }
        public bool IsEnabled { get; set; }
        public int DeviceId { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }


        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public ControlNetModelConfig ControlNetConfig { get; set; }
    }
}
