using Microsoft.ML.OnnxRuntime;
using OnnxStack.Common.Config;
using OnnxStack.Core.Config;
using OnnxStack.UI.Views;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text.Json.Serialization;

namespace OnnxStack.UI.Models
{
    public class OnnxStackUIConfig : IConfigSection
    {
        public ModelCacheMode ModelCacheMode { get; set; }

        public bool ImageAutoSave { get; set; }
        public bool ImageAutoSaveBlueprint { get; set; }
        public string ImageAutoSaveDirectory { get; set; }
        public int RealtimeRefreshRate { get; set; } = 100;
        public bool RealtimeHistoryEnabled { get; set; }
        public int DefaultDeviceId { get; set; }
        public int DefaultInterOpNumThreads { get; set; }
        public int DefaultIntraOpNumThreads { get; set; }
        public ExecutionMode DefaultExecutionMode { get; set; }
        public ExecutionProvider DefaultExecutionProvider { get; set; }
        public ObservableCollection<UpscaleModelSetViewModel> UpscaleModelSets { get; set; } = new ObservableCollection<UpscaleModelSetViewModel>();
        public ObservableCollection<StableDiffusionModelSetViewModel> StableDiffusionModelSets { get; set; } = new ObservableCollection<StableDiffusionModelSetViewModel>();

        public List<ModelConfigTemplate> ModelTemplates { get; set; } = new List<ModelConfigTemplate>();

        public void Initialize()
        {
        }



        public ObservableCollection<ModelTemplateViewModel> Templates { get; set; } = new ObservableCollection<ModelTemplateViewModel>();
    }

    public enum ModelCacheMode
    {
        Single = 0,
        Multiple = 1
    }
}
