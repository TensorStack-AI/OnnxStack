using OnnxStack.Core.Image;

namespace OnnxStack.Core.Video
{
    public record VideoFrame(byte[] Frame)
    {
        public OnnxImage ExtraFrame { get; set; }
    }
}
