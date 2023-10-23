using System.Collections.Generic;

namespace OnnxStack.Core.Config
{
    public interface IOnnxModelSetConfig : IOnnxModel
    {
        List<OnnxModelSessionConfig> ModelConfigurations { get; set; }
    }
}
