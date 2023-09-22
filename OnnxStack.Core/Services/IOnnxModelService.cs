using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System;
using System.Collections.Generic;

namespace OnnxStack.Core.Services
{
    public interface IOnnxModelService : IDisposable
    {
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInference(OnnxModelType modelType, IReadOnlyCollection<NamedOnnxValue> inputs);
    }
}