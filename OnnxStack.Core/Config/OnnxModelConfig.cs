using Microsoft.ML.OnnxRuntime;
using System.Text.Json.Serialization;

namespace OnnxStack.Core.Config
{
    public record OnnxModelConfig
    {
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

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public OnnxModelPrecision? Precision { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int RequiredMemory { get; set; }
    }
}
