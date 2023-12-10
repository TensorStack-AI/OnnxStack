using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace OnnxStack.Core.Config
{
    public interface IOnnxModelSetConfig : IOnnxModel
    {
        bool IsEnabled { get; set; }
        int DeviceId { get; set; }
        int InterOpNumThreads { get; set; }
        int IntraOpNumThreads { get; set; }
        ExecutionMode ExecutionMode { get; set; }
        ExecutionProvider ExecutionProvider { get; set; }
        List<OnnxModelConfig> ModelConfigurations { get; set; }
    }
}
