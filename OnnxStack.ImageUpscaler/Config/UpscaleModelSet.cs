using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Config
{
    public class UpscaleModelSet : IOnnxModelSetConfig
    {
        public string Name { get; set; }
        public int Channels { get; set; }
        public int SampleSize { get; set; }
        public int ScaleFactor { get; set; }
        public bool IsEnabled { get; set; }
        public int DeviceId { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public ExecutionProvider ExecutionProvider { get; set; }
        public List<OnnxModelConfig> ModelConfigurations { get; set; }
      }
}
