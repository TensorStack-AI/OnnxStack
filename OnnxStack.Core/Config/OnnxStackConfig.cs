using OnnxStack.Common.Config;

namespace OnnxStack.Core.Config
{
    public class OnnxStackConfig : IConfigSection
    {
        public string TempPath { get; set; } = ".temp";
        public string FFmpegPath { get; set; } = "ffmpeg.exe";
        public string FFprobePath { get; set; } = "ffprobe.exe";

        public void Initialize()
        {
        }
    }
}
