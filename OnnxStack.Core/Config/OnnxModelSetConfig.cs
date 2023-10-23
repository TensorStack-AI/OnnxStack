using System.Collections.Generic;

namespace OnnxStack.Core.Config
{
    public class OnnxModelSetConfig : IOnnxModelSetConfig
    {
        public string Name { get; set; }
        public List<OnnxModelSessionConfig> ModelConfigurations { get; set; }
    }
}
