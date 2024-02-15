using System;

namespace OnnxStack.Core.Video
{
    public sealed record VideoInfo(TimeSpan Duration, float FrameRate)
    {
        public VideoInfo(int height, int width, TimeSpan duration, float frameRate) : this(duration, frameRate)
        {
            Height = height;
            Width = width;
        }
        public int Height { get; set; }
        public int Width { get; set; }
    }
}
