using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;

namespace OnnxStack.Core
{
    internal static class Extensions
    {
        public static SessionOptions GetSessionOptions(this OnnxStackConfig configuration)
        {
            var sessionOptions = new SessionOptions();
            switch (configuration.ExecutionProviderTarget)
            {
                case ExecutionProvider.DirectML:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.EnableMemoryPattern = false;
                    sessionOptions.AppendExecutionProvider_DML(configuration.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                case ExecutionProvider.Cpu:
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                default:
                case ExecutionProvider.Cuda:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.AppendExecutionProvider_CUDA(configuration.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
            }
        }

    }
}
