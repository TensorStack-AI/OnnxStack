using OnnxStack.Common.Config;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using System.Collections.Generic;

namespace OnnxStack.ImageUpscaler.Config
{
    public class ImageUpscalerConfig : IConfigSection
    {
        public List<UpscaleModelSet> ModelSets { get; set; } = new List<UpscaleModelSet>();

        public void Initialize()
        {
        }
    }
}
