using OnnxStack.Core.Model;

namespace OnnxStack.Core.Config
{
    public interface IOnnxModelSetConfig : IOnnxModel
    {
        OnnxExecutionProvider ExecutionProvider { get; set; }
    }
}
