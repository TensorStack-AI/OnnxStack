using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace OnnxStack.Core.Config
{
    public class OnnxModelSetConfig : IOnnxModelSetConfig
    {
        public string Name { get; set; }
        public bool IsEnabled { get; set; }
        public int DeviceId { get; set; }
        public string OnnxModelPath { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }
        public List<OnnxModelSessionConfig> ModelConfigurations { get; set; }
    }
}
