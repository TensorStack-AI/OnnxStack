using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System.Text.Json.Serialization;

namespace OnnxStack.ImageUpscaler.Common
{
    public record UpscaleModelSet : IOnnxModelSetConfig
    {
        public string Name { get; set; }
        public bool IsEnabled { get; set; }
        public int DeviceId { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }


        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public UpscaleModelConfig UpscaleModelConfig { get; set; }

    }
}
