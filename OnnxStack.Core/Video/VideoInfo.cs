using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

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

        public double AspectRatio => (double)Width / Height;
    }

    public record VideoMetadata
    {
        [JsonPropertyName("format")]
        public VideoFormat Format { get; set; }

        [JsonPropertyName("streams")]
        public List<VideoStream> Streams { get; set; }
    }

    [JsonNumberHandling(JsonNumberHandling.AllowReadingFromString)]
    public record VideoFormat
    {
        [JsonPropertyName("filename")]
        public string FileName { get; set; }

        [JsonPropertyName("nb_streams")]
        public int StreamCount { get; set; }

        [JsonPropertyName("format_name")]
        public string FormatName { get; set; }

        [JsonPropertyName("format_long_name")]
        public string FormatLongName { get; set; }

        [JsonPropertyName("size")]
        public long Size { get; set; }

        [JsonPropertyName("bit_rate")]
        public long BitRate { get; set; }
    }

    [JsonNumberHandling(JsonNumberHandling.AllowReadingFromString)]
    public record VideoStream
    {
        [JsonPropertyName("codec_type")]
        public string Type { get; set; }

        [JsonPropertyName("codec_name")]
        public string CodecName { get; set; }

        [JsonPropertyName("codec_long_name")]
        public string CodecLongName { get; set; }

        [JsonPropertyName("pix_fmt")]
        public string PixelFormat { get; set; }

        [JsonPropertyName("width")]
        public int Width { get; set; }

        [JsonPropertyName("height")]
        public int Height { get; set; }

        [JsonPropertyName("nb_frames")]
        public int FrameCount { get; set; }

        [JsonPropertyName("duration")]
        public float DurationSeconds { get; set; }

        public float FramesPerSecond => GetFramesPerSecond();

        public TimeSpan Duration => TimeSpan.FromSeconds(DurationSeconds);

        private float GetFramesPerSecond()
        {
            if (FrameCount == 0 || DurationSeconds == 0)
                return 0;

            var framesPerSec = FrameCount / DurationSeconds;
            if (framesPerSec < 1)
                return MathF.Round(framesPerSec, 2);

            return MathF.Round(framesPerSec, 0);
        }
    }
}
