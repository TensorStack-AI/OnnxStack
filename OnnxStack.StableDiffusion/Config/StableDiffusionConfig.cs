using OnnxStack.Common.Config;
using System.Collections.Generic;

namespace OnnxStack.StableDiffusion.Config
{
    public class StableDiffusionConfig : IConfigSection
    {
        public List<StableDiffusionModelSet> ModelSets { get; set; } = new List<StableDiffusionModelSet>();

        public void Initialize()
        {
        }
    }
}
