using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Enums;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Config
{
    public record ControlNetModelSet : IOnnxModelSetConfig
    {
        public ControlNetType Type { get; set; }
        public string Name { get; set; }
        public bool IsEnabled { get; set; }
        public int DeviceId { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }
        public List<OnnxModelConfig> ModelConfigurations { get; set; }
    }
}
