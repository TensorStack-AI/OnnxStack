using System;

namespace OnnxStack.Core.Video
{
    public record VideoInfo(int Width, int Height, TimeSpan Duration, int FPS);
}
