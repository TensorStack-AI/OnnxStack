using OnnxStack.Common.Config;

namespace OnnxStack.Core.Config
{
    public class OnnxStackConfig : IConfigSection
    {
        public string TempPath { get; set; } = ".temp";
        public string VideoCodec { get; set; } = "mp4v";

        public void Initialize()
        {
        }
    }
}
