using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace OnnxStack.Core.Config
{
    public interface IOnnxModelSetConfig : IOnnxModel
    {
        public int DeviceId { get; set; }
        public string OnnxModelPath { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }
        List<OnnxModelSessionConfig> ModelConfigurations { get; set; }
    }
}
