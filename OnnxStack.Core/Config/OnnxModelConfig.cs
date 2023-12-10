using Microsoft.ML.OnnxRuntime;
using System.Text.Json.Serialization;

namespace OnnxStack.Core.Config
{
    public class OnnxModelConfig
    {
        public OnnxModelType Type { get; set; }
        public string OnnxModelPath { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public int? DeviceId { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public int? InterOpNumThreads { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public int? IntraOpNumThreads { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public ExecutionMode? ExecutionMode { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public ExecutionProvider? ExecutionProvider { get; set; }
    }
}
