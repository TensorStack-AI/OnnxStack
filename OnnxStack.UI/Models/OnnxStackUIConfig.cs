using Microsoft.ML.OnnxRuntime;
using OnnxStack.Common.Config;
using OnnxStack.Core.Config;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace OnnxStack.UI.Models
{
    public class OnnxStackUIConfig : IConfigSection
    {
        public int DefaultDeviceId { get; set; }
        public int DefaultInterOpNumThreads { get; set; }
        public int DefaultIntraOpNumThreads { get; set; }
        public ExecutionMode DefaultExecutionMode { get; set; }
        public ExecutionProvider DefaultExecutionProvider { get; set; }
        public ExecutionProvider SupportedExecutionProvider => GetSupportedExecutionProvider();
        public ObservableCollection<UpscaleModelSetViewModel> UpscaleModelSets { get; set; } = new ObservableCollection<UpscaleModelSetViewModel>();
        public ObservableCollection<StableDiffusionModelSetViewModel> StableDiffusionModelSets { get; set; } = new ObservableCollection<StableDiffusionModelSetViewModel>();
        public ObservableCollection<ControlNetModelSetViewModel> ControlNetModelSets { get; set; } = new ObservableCollection<ControlNetModelSetViewModel>();

        public ExecutionProvider GetSupportedExecutionProvider()
        {
#if DEBUG_CUDA || RELEASE_CUDA
            return ExecutionProvider.Cuda;
#elif DEBUG_TENSORRT || RELEASE_TENSORRT
            return ExecutionProvider.TensorRT;
#else
            return ExecutionProvider.DirectML;
#endif
        }

        public void Initialize()
        {
            DefaultExecutionProvider = DefaultExecutionProvider == SupportedExecutionProvider || DefaultExecutionProvider == ExecutionProvider.Cpu
                ? DefaultExecutionProvider
                : SupportedExecutionProvider;
        }

    }
}
