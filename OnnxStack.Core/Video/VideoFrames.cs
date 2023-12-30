using System.Collections.Generic;

namespace OnnxStack.Core.Video
{
    public record VideoFrames(float FPS, List<byte[]> Frames);
}
