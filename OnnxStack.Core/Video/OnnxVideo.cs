using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.Core.Video
{
    public sealed class OnnxVideo : IDisposable
    {
        private readonly VideoInfo _info;
        private readonly IReadOnlyList<OnnxImage> _frames;

        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxVideo"/> class.
        /// </summary>
        /// <param name="info">The information.</param>
        /// <param name="frames">The frames.</param>
        public OnnxVideo(VideoInfo info, List<OnnxImage> frames)
        {
            _info = info;
            _frames = frames;
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxVideo"/> class.
        /// </summary>
        /// <param name="info">The information.</param>
        /// <param name="videoTensor">The video tensor.</param>
        public OnnxVideo(VideoInfo info, DenseTensor<float> videoTensor)
        {
            _info = info;
            _frames = videoTensor
                .SplitBatch()
                .Select(x => new OnnxImage(x))
                .ToList();
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxVideo"/> class.
        /// </summary>
        /// <param name="info">The information.</param>
        /// <param name="videoTensors">The video tensors.</param>
        public OnnxVideo(VideoInfo info, IEnumerable<DenseTensor<float>> videoTensors)
        {
            _info = info;
            _frames = videoTensors
                .Select(x => new OnnxImage(x))
                .ToList();
        }


        /// <summary>
        /// Gets the height.
        /// </summary>
        public int Height => _info.Height;

        /// <summary>
        /// Gets the width.
        /// </summary>
        public int Width => _info.Width;

        /// <summary>
        /// Gets the frame rate.
        /// </summary>
        public float FrameRate => _info.FrameRate;

        /// <summary>
        /// Gets the duration.
        /// </summary>
        public TimeSpan Duration => _info.Duration;

        /// <summary>
        /// Gets the information.
        /// </summary>
        public VideoInfo Info => _info;

        /// <summary>
        /// Gets the frames.
        /// </summary>
        public IReadOnlyList<OnnxImage> Frames => _frames;

        /// <summary>
        /// Gets the aspect ratio.
        /// </summary>
        public double AspectRatio => (double)_info.Width / _info.Height;

        /// <summary>
        /// Gets a value indicating whether this instance has video.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance has video; otherwise, <c>false</c>.
        /// </value>
        public bool HasVideo
        {
            get { return !_frames.IsNullOrEmpty(); }
        }


        /// <summary>
        /// Gets the frame at the specified index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns></returns>
        public OnnxImage GetFrame(int index)
        {
            if (_frames?.Count > index)
                return _frames[index];

            return null;
        }


        /// <summary>
        /// Resizes the video.
        /// </summary>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        public void Resize(int height, int width)
        {
            foreach (var frame in _frames)
                frame.Resize(height, width);

            _info.Width = width;
            _info.Height = height;
        }


        /// <summary>
        /// Saves the video to file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public Task SaveAsync(string filename, CancellationToken cancellationToken = default)
        {
            return VideoHelper.WriteVideoFramesAsync(this, filename, cancellationToken);
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            foreach (var item in _frames)
            {
                item?.Dispose();
            }
        }

        /// <summary>
        /// Load a video from file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public static async Task<OnnxVideo> FromFileAsync(string filename, float? frameRate = default, CancellationToken cancellationToken = default)
        {
            var videoBytes = await File.ReadAllBytesAsync(filename, cancellationToken);
            var videoInfo = await VideoHelper.ReadVideoInfoAsync(videoBytes);
            if (frameRate.HasValue)
                videoInfo = videoInfo with { FrameRate = Math.Min(videoInfo.FrameRate, frameRate.Value) };

            var videoFrames = await VideoHelper.ReadVideoFramesAsync(videoBytes, videoInfo.FrameRate, cancellationToken);
            return new OnnxVideo(videoInfo, videoFrames);
        }
    }
}
