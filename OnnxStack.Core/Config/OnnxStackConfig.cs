using OnnxStack.Common.Config;
using System.Collections.Generic;

namespace OnnxStack.Core.Config
{
    public class OnnxStackConfig : IConfigSection
    {
        public List<OnnxModelSetConfig> OnnxModelSets { get; set; } = new List<OnnxModelSetConfig>();

        public void Initialize()
        {
        }
    }
}
