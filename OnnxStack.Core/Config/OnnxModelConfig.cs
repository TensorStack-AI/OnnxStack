using OnnxStack.Core.Model;

namespace OnnxStack.Core.Config
{
    public record OnnxModelConfig
    {
        public string OnnxModelPath { get; set; }
        public bool IsOptimizationSupported { get; set; } = true;
        public OnnxExecutionProvider ExecutionProvider { get; set; }
    }
}
