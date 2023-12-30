using OnnxStack.Core.Video;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.Core.Services
{
    /// <summary>
    /// Service with basic handling of video for use in OnnxStack, Frame->Video and Video->Frames
    /// </summary>
    public interface IVideoService
    {
        /// <summary>
        /// Gets the video information asynchronous.
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<VideoInfo> GetVideoInfoAsync(byte[] videoBytes, CancellationToken cancellationToken = default);

        /// <summary>
        /// Gets the video information asynchronous.
        /// </summary>
        /// <param name="videoStream">The video stream.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<VideoInfo> GetVideoInfoAsync(Stream videoStream, CancellationToken cancellationToken = default);

        /// <summary>
        /// Gets the video information, Size, FPS, Duration etc.
        /// </summary>
        /// <param name="videoInput">The video input.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException">No video data found</exception>
        Task<VideoInfo> GetVideoInfoAsync(VideoInput videoInput, CancellationToken cancellationToken = default);


        /// <summary>
        /// Creates a collection of PNG frames from a video source
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <param name="videoFPS">The video FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<VideoFrames> CreateFramesAsync(byte[] videoBytes, float videoFPS, CancellationToken cancellationToken = default);


        /// <summary>
        /// Creates a collection of PNG frames from a video source
        /// </summary>
        /// <param name="videoStream">The video stream.</param>
        /// <param name="videoFPS">The video FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<VideoFrames> CreateFramesAsync(Stream videoStream, float videoFPS, CancellationToken cancellationToken = default);


        /// <summary>
        /// Creates a collection of PNG frames from a video source
        /// </summary>
        /// <param name="videoInput">The video input.</param>
        /// <param name="videoFPS">The video FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException">VideoTensor not supported</exception>
        /// <exception cref="ArgumentException">No video data found</exception>
        Task<VideoFrames> CreateFramesAsync(VideoInput videoInput, float videoFPS, CancellationToken cancellationToken = default);


        /// <summary>
        /// Creates and MP4 video from a collection of PNG images.
        /// </summary>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="videoFPS">The video FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<VideoResult> CreateVideoAsync(IEnumerable<byte[]> videoFrames, float videoFPS, CancellationToken cancellationToken = default);


        /// <summary>
        /// Creates and MP4 video from a collection of PNG images.
        /// </summary>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        Task<VideoResult> CreateVideoAsync(VideoFrames videoFrames, CancellationToken cancellationToken = default);


        /// <summary>
        /// Streams frames as PNG as they are processed from a video source
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <param name="targetFPS">The target FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<byte[]> StreamFramesAsync(byte[] videoBytes, float targetFPS, CancellationToken cancellationToken = default);
    }
}