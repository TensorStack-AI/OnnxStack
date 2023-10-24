using OnnxStack.Common.Config;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.Core.Config
{
    public class OnnxStackConfig : IConfigSection
    {
        public List<OnnxModelSetConfig> OnnxModelSets { get; set; } = new List<OnnxModelSetConfig>();

        public void Initialize()
        {
            if (OnnxModelSets.IsNullOrEmpty())
                return;

            foreach (var modelSet in OnnxModelSets)
            {
                modelSet.ApplyConfigurationOverrides();
            }
        }
    }
}
