using OnnxStack.Core.Image;

namespace OnnxStack.Core.Video
{
    public record VideoFrame(byte[] Frame, InputImage ControlImage = default);
}
