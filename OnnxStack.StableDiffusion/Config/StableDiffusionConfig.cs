using OnnxStack.Common.Config;
using OnnxStack.Core;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace OnnxStack.StableDiffusion.Config
{
    public class StableDiffusionConfig : IConfigSection
    {
        public List<ModelOptions> OnnxModelSets { get; set; } = new List<ModelOptions>();

        public void Initialize()
        {
            if (OnnxModelSets.IsNullOrEmpty())
                return;

            foreach (var modelSet in OnnxModelSets)
            {
                modelSet.ApplyConfigurationOverrides();
                modelSet.BlankTokenValueArray = Enumerable.Repeat(modelSet.BlankTokenId, modelSet.InputTokenLimit).ToImmutableArray();
            }
        }
    }
}
