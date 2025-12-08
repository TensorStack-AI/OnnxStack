using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using SixLabors.ImageSharp;
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
        private OnnxImage[] _frames;
        private int _frameCount;
        private int _height;
        private int _width;
        private float _frameRate;
        private TimeSpan _duration;
        private double _aspectRatio;


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxVideo"/> class.
        /// </summary>
        /// <param name="frames">The frame count</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public OnnxVideo(int frameCount, float frameRate, int width, int height)
        {
            Initialize(frameCount, frameRate, width, height);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxVideo"/> class.
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="frameRate">The frame rate.</param>
        public OnnxVideo(OnnxImage[] imageFrames, float frameRate)
        {
            InitializeFrames(imageFrames, frameRate);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxVideo"/> class.
        /// </summary>
        /// <param name="info">The information.</param>
        /// <param name="frames">The frames.</param>
        public OnnxVideo(IEnumerable<OnnxImage> imageFrames, float frameRate)
        {
            var frames = imageFrames.ToArray();
            InitializeFrames(frames, frameRate);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxVideo"/> class.
        /// </summary>
        /// <param name="info">The information.</param>
        /// <param name="videoTensor">The video tensor.</param>
        public OnnxVideo(DenseTensor<float> videoTensor, float frameRate, ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne)
        {
            var frames = videoTensor
                .SplitBatch()
                .Select(x => new OnnxImage(x, normalizeType))
                .ToArray();
            InitializeFrames(frames, frameRate);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="OnnxVideo"/> class.
        /// </summary>
        /// <param name="info">The information.</param>
        /// <param name="videoTensors">The video frame tensors.</param>
        public OnnxVideo(List<DenseTensor<float>> frameTensors, float frameRate, ImageNormalizeType normalizeType = ImageNormalizeType.OneToOne)
        {
            var frames = new OnnxImage[frameTensors.Count];
            Parallel.For(0, frameTensors.Count, index =>
            {
                frames[index] = new OnnxImage(frameTensors[index], normalizeType);
            });
            InitializeFrames(frames, frameRate);
        }


        /// <summary>
        /// Gets the height.
        /// </summary>
        public int Height => _height;

        /// <summary>
        /// Gets the width.
        /// </summary>
        public int Width => _width;

        /// <summary>
        /// Gets the frame rate.
        /// </summary>
        public float FrameRate => _frameRate;

        /// <summary>
        /// Gets the duration.
        /// </summary>
        public TimeSpan Duration => _duration;

        /// <summary>
        /// Gets the frame count.
        /// </summary>
        public int FrameCount => _frameCount;

        /// <summary>
        /// Gets the frames.
        /// </summary>
        public IReadOnlyList<OnnxImage> Frames => _frames;

        /// <summary>
        /// Gets the aspect ratio.
        /// </summary>
        public double AspectRatio => _aspectRatio;

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
            if (_frames?.Length > index)
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

            _width = width;
            _height = height;
        }


        /// <summary>
        /// Normalizes the frame brightness.
        /// </summary>
        public void NormalizeBrightness()
        {
            var averageBrightness = _frames.Average(x => x.GetBrightness());
            foreach (var frame in _frames)
            {
                var frameBrightness = frame.GetBrightness();
                var adjustmentFactor = averageBrightness / frameBrightness;
                frame.SetBrightness(adjustmentFactor);
            }
        }


        /// <summary>
        /// Repeats the specified count.
        /// </summary>
        /// <param name="count">The count.</param>
        public void Repeat(int count)
        {
            var frames = new List<OnnxImage>();
            for (int i = 0; i < count; i++)
            {
                frames.AddRange(_frames);
            }
            InitializeFrames(frames.ToArray(), _frameRate);
        }


        /// <summary>
        /// Saves the video to file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public Task SaveAsync(string filename, CancellationToken cancellationToken = default)
        {
            return VideoHelper.WriteVideoAsync(filename, this, cancellationToken);
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
        /// Initializes video dimensions.
        /// </summary>
        /// <param name="frames">The frames.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        private void Initialize(int frameCount, float frameRate, int width, int height)
        {
            _width = width;
            _height = height;
            _frameRate = frameRate;
            _frameCount = frameCount;
            _aspectRatio = (double)_width / _height;
            _duration = TimeSpan.FromSeconds(frameCount / _frameRate);
        }


        /// <summary>
        /// Initializes the frames.
        /// </summary>
        /// <param name="frames">The frames.</param>
        /// <param name="frameRate">The frame rate.</param>
        private void InitializeFrames(OnnxImage[] frames, float frameRate)
        {
            _frames = frames;
            var firstFrame = _frames[0];
            Initialize(_frames.Length, frameRate, firstFrame.Width, firstFrame.Height);
        }


        /// <summary>
        /// Load a video from file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="frameRate">The desired frame rate.</param>
        /// <param name="width">The desired width. (aspect will be preserved)</param>
        /// <param name="height">The desired height. (aspect will be preserved)</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public static async Task<OnnxVideo> FromFileAsync(string filename, float? frameRate = default, int? width = default, int? height = default, CancellationToken cancellationToken = default)
        {
            var videoBytes = await File.ReadAllBytesAsync(filename, cancellationToken);
            return await FromBytesAsync(videoBytes, frameRate, width, height, cancellationToken);
        }


        /// <summary>
        /// Load a video from bytes
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <param name="frameRate">The desired frame rate.</param>
        /// <param name="width">The desired width. (aspect will be preserved)</param>
        /// <param name="height">The desired height. (aspect will be preserved)</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;OnnxVideo&gt; representing the asynchronous operation.</returns>
        public static async Task<OnnxVideo> FromBytesAsync(byte[] videoBytes, float? frameRate = default, int? width = default, int? height = default, CancellationToken cancellationToken = default)
        {
            return await Task.Run(() => VideoHelper.ReadVideoAsync(videoBytes, frameRate, width, height, cancellationToken));
        }

    }
}
