using Microsoft.ML.OnnxRuntime;

namespace OnnxStack.Core.Config
{
    public class OnnxModelSessionConfig
    {
        public OnnxModelType Type { get; set; }
        public bool IsDisabled { get; set; }
        public int DeviceId { get; set; }
        public string OnnxModelPath { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }
    }
}
