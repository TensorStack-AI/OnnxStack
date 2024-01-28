using Microsoft.ML.OnnxRuntime;

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
    }
}
