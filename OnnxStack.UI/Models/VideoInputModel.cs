using OnnxStack.Core.Video;
using System.Text.Json.Serialization;

namespace OnnxStack.UI.Models
{
    public class VideoInputModel
    {
        public VideoInfo VideoInfo { get; set; }
        public string FileName { get; set; }
        public byte[] VideoBytes { get; set; }
    }

}