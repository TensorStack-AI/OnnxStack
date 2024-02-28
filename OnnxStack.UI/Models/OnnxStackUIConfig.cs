using Microsoft.ML.OnnxRuntime;
using OnnxStack.Common.Config;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Enums;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text.Json.Serialization;

namespace OnnxStack.UI.Models
{
    public class OnnxStackUIConfig : IConfigSection
    {
        public int DefaultDeviceId { get; set; }
        public int DefaultInterOpNumThreads { get; set; }
        public int DefaultIntraOpNumThreads { get; set; }
        public ExecutionMode DefaultExecutionMode { get; set; }
        public ExecutionProvider DefaultExecutionProvider { get; set; }
        public MemoryModeType DefaultMemoryMode { get; set; }

        [JsonIgnore]
        public ExecutionProvider SupportedExecutionProvider => GetSupportedExecutionProvider();
        public ObservableCollection<UpscaleModelSetViewModel> UpscaleModelSets { get; set; } = new ObservableCollection<UpscaleModelSetViewModel>();
        public ObservableCollection<StableDiffusionModelSetViewModel> StableDiffusionModelSets { get; set; } = new ObservableCollection<StableDiffusionModelSetViewModel>();
        public ObservableCollection<ControlNetModelSetViewModel> ControlNetModelSets { get; set; } = new ObservableCollection<ControlNetModelSetViewModel>();
        public ObservableCollection<FeatureExtractorModelSetViewModel> FeatureExtractorModelSets { get; set; } = new ObservableCollection<FeatureExtractorModelSetViewModel>();

        public ExecutionProvider GetSupportedExecutionProvider()
        {
#if DEBUG_NVIDIA || RELEASE_NVIDIA
            return ExecutionProvider.Cuda;
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
