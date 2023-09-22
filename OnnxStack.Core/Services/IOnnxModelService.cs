using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System.Collections.Generic;

namespace OnnxStack.Core.Services
{
    public interface IOnnxModelService
    {
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInference(OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs);
    }
}