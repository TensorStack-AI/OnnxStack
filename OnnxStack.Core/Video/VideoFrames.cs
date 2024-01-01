using System.Collections.Generic;

namespace OnnxStack.Core.Video
{
    public record VideoFrames(VideoInfo Info, IReadOnlyList<byte[]> Frames);
}
