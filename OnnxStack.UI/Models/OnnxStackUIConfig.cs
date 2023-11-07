using OnnxStack.Common.Config;
using System.Collections.Generic;

namespace OnnxStack.UI.Views
{
    public class OnnxStackUIConfig : IConfigSection
    {
        public List<ModelConfigTemplate> ModelTemplates { get; set; } = new List<ModelConfigTemplate>();

        public void Initialize()
        {
        }
    }
}
